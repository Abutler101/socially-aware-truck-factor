import math
import os
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Dict, Callable, Any, Optional

import docker
import numpy as np
from loguru import logger
from git import Repo
import networkx as nx
from tree_sitter_languages import get_parser
from tree_sitter import Parser, Tree, Node
from fuzzywuzzy import fuzz

from analysis.estimators.base_estimator import TruckFactorEstimator, EstimatorConfig
from shared_models.analysis_target import AnalysisTarget
from analysis.shared_utils.third_party_files import load_path_regexes
from analysis.shared_utils.graph_measures import weighted_deg_cent, weighted_in_deg_cent, weighted_out_deg_cent
from shared_models.commit import CommitSM

SCRATCH_PATH = Path(__file__).parents[1].joinpath("output/scratch")
OUTPUT_PATH = Path(__file__).parents[1].joinpath("output/import_graphs")


class FileImportanceMetric(str, Enum):
    PAGE_RANK = "PageRank"
    IN_DEG = "InDegree"
    OUT_DEG = "OutDegree"
    BET_CENT = "BetweenessCentrality"
    DEG_CENT = "DegreeCentrality"


class SupportedLanguage(str, Enum):
    PYTHON = "Python"
    JAVA = "Java"
    CPP = "C++"
    RUBY = "Ruby"
    JS = "JavaScript"
    PHP = "PHP"
    UNSUPPORTED = "Unsupported"

    @classmethod
    def _missing_(cls, value):
        logger.warning(f"{value} is not a supported Language")
        return cls.UNSUPPORTED


class HaratianConfig(EstimatorConfig):
    file_importance_metric: FileImportanceMetric = FileImportanceMetric.BET_CENT

    base_doa: float = 3.293
    first_authorship_weight: float = 1.098
    num_changes_weight: float = 0.164
    others_change_count_weight: float = 0.321

    doa_norm_threshold: float = 0.75
    doa_abs_threshold: float = 3.293
    coverage_threshold: float = 0.5


class HaratianEstimator(TruckFactorEstimator):
    config: HaratianConfig
    target: AnalysisTarget
    _third_party_detection_regex: re.Pattern

    _IMPORTANCE_METRIC_MAP: Dict[FileImportanceMetric, Callable[[Any, str], Dict[str, float]]] = dict()
    _CLONE_URL_TEMPLATE: str = "https://github.com/{owner}/{name}.git"
    _PARSER_MAP: Dict[SupportedLanguage, Parser]
    _EXTENSION_MAP: Dict[SupportedLanguage, List[str]]

    def __init__(self, config: HaratianConfig):
        self.config = config
        self._IMPORTANCE_METRIC_MAP = {
            FileImportanceMetric.PAGE_RANK: nx.pagerank,
            FileImportanceMetric.IN_DEG: weighted_in_deg_cent,
            FileImportanceMetric.OUT_DEG: weighted_out_deg_cent,
            FileImportanceMetric.BET_CENT: nx.betweenness_centrality,
            FileImportanceMetric.DEG_CENT: weighted_deg_cent,
        }
        self._PARSER_MAP = {
            SupportedLanguage.PYTHON: get_parser("python"),
            SupportedLanguage.JAVA: get_parser("java"),
            SupportedLanguage.CPP: get_parser("cpp"),
            SupportedLanguage.RUBY: get_parser("ruby"),
            SupportedLanguage.JS: get_parser("javascript"),
            SupportedLanguage.PHP: get_parser("php")
        }
        self._EXTENSION_MAP = {
            SupportedLanguage.PYTHON: [".py"],
            SupportedLanguage.JAVA: [".java"],
            SupportedLanguage.CPP: [".cpp", ".cc", ".h"],
            SupportedLanguage.RUBY: [".rb"],
            SupportedLanguage.JS: [".js"],
            SupportedLanguage.PHP: [".php"]
        }
        self._third_party_detection_regex = load_path_regexes()
        SCRATCH_PATH.mkdir(exist_ok=True)
        OUTPUT_PATH.mkdir(exist_ok=True)

    def load_project(self, project_data: AnalysisTarget):
        self.target = project_data

    def run_estimation(self) -> Tuple[int, List[str]]:
        truck_factor = 0
        truck_factor_contributors = []
        # Internal Dep Graph Gen is expensive, see if its already been done
        graph_path = OUTPUT_PATH.joinpath(f"{self.target.repo_name}.gml")
        if graph_path.exists():
            project_internal_dep_graph = nx.read_gml(graph_path)
        else:
            project_internal_dep_graph = self._build_internal_dep_graph()
            if project_internal_dep_graph is not None:
                nx.write_gml(project_internal_dep_graph, graph_path)

        # If Graph Construction failed, skip Haratian Estimator
        if project_internal_dep_graph is None:
            return 0, []

        # Calculate File Importance scores
        file_importance_scores = self._calculate_file_importance_scores(project_internal_dep_graph)

        # Do normal Avelino Estimator stuff
        file_author_map = self._build_author_map()
        # Eliminate 3rd party files from importance scores
        file_importance_scores = {
            file: score for file, score in file_importance_scores.items() if file in file_author_map
        }
        author_file_map = self._invert_file_author_map(file_author_map)
        # sort authors by total file importance
        authors_total_importance = []
        for author, files in author_file_map.items():
            total_importance = sum([file_importance_scores[file] for file in files])
            authors_total_importance.append((author, total_importance))
        authors_ranked_by_file_importance=sorted(authors_total_importance, key=lambda e: e[1], reverse=1)
        # calc starting coverage
        coverage = self._calc_coverage(file_author_map, file_importance_scores)
        while coverage > self.config.coverage_threshold:
            top_author = authors_ranked_by_file_importance[truck_factor][0]
            self._pop_author_from_map(file_author_map, top_author)
            truck_factor_contributors.append(top_author)
            truck_factor += 1
            coverage = self._calc_coverage(file_author_map, file_importance_scores)

        return truck_factor, truck_factor_contributors

    def _build_internal_dep_graph(self) -> Optional[nx.DiGraph]:
        """
        Using Tree-Sitter construct the file->file reference graph for the target project
        """
        dep_graph = nx.DiGraph()
        dep_graph.add_nodes_from(self.target.known_files)
        # Clone Repo to scratch Dir
        repo_path = self._clone_repo(self.target.repo_name)
        # Detect Language
        language = self._detect_repo_lang(repo_path)
        if language is SupportedLanguage.UNSUPPORTED:
            logger.warning(f"Haratian Estimator Skipped for {self.target.repo_name} due to unsupported primary lang")
            return None
        # Setup Tree Sitter
        parser = self._PARSER_MAP[language]
        # Collect local imports of each file and store in Graph
        for root_path, _, files in os.walk(repo_path):
            root_path = Path(root_path)
            files = [Path(file) for file in files]
            for file in files:
                abs_file_path = root_path.joinpath(file)
                if abs_file_path.suffix not in self._EXTENSION_MAP[language]:
                    continue
                source = ""
                try:
                    with abs_file_path.open("r") as file_pointer:
                        source = file_pointer.read()
                except UnicodeDecodeError as err:
                    if source == "":
                        continue
                tree = parser.parse(bytes(source, 'utf8'))
                import_nodes = self._find_import_nodes(tree, language)
                imported_paths = self._extract_paths_from_nodes(import_nodes, repo_path, abs_file_path, language)

                # Convert abs paths to relative paths that match with known files in target
                importer_path_str = self._map_to_known_file(repo_path, abs_file_path)
                if importer_path_str is None:
                    continue
                # Keys are relative paths, vals are number of times imported
                relative_imported_paths: Dict[str, int] = dict()
                for abs_imported_path in imported_paths:
                    # Map to known relative path
                    relative_path = self._map_to_known_file(repo_path, abs_imported_path)
                    if relative_path is None:
                        continue

                    if relative_path in relative_imported_paths:
                        relative_imported_paths[relative_path] += 1
                    else:
                        relative_imported_paths[relative_path] = 1
                # Add Edges from the current file to the locally imported ones
                for relative_imported_path, import_count in relative_imported_paths.items():
                    weight = import_count
                    dep_graph.add_edge(importer_path_str, relative_imported_path, weight=weight)
        return dep_graph

    def _clone_repo(self, internal_name_rep: str):
        repo_owner = "-".join(internal_name_rep.split("-")[:-1])
        repo_name = internal_name_rep.split("-")[-1]
        clone_url = self._CLONE_URL_TEMPLATE.format(owner=repo_owner, name=repo_name)
        clone_path = SCRATCH_PATH.joinpath(internal_name_rep)
        shutil.rmtree(clone_path, ignore_errors=True)
        logger.debug(f"trying to clone {clone_url}")
        repo = Repo.clone_from(url=clone_url, to_path=clone_path)
        repo.git.checkout(self.target.commits[0].commit_hash)
        return clone_path

    def _detect_repo_lang(self, repo_path: Path) -> SupportedLanguage:
        docker_client = docker.from_env()
        direct_out = docker_client.containers.run(
            image="crazymax/linguist:latest",
            stderr=True,
            stdout=True,
            remove=True,
            volumes={repo_path: {"bind": "/repo"}}
        )
        detected_langs = direct_out.decode("utf-8").split("\n")
        logger.info(f"Primary lang detected: {detected_langs[0].split(' ')[-1]}")
        lang_string = detected_langs[0].split(" ")[-1]
        return SupportedLanguage(lang_string)

    def _find_import_nodes(self, tree: Tree,language: SupportedLanguage) -> List[Node]:
        # First is parent node type,
        # second is list of acceptable parent types,
        # third is list of lists of required child types
        IMPORT_TYPES_MAP = {
            SupportedLanguage.PYTHON: (["import_statement", "import_from_statement"], [], []),
            SupportedLanguage.JAVA: (["import_declaration"], [],[]),
            SupportedLanguage.CPP: (["preproc_include"], [], []),
            SupportedLanguage.RUBY: (["call"], ["program"], [["identifier", "argument_list"]]),
            SupportedLanguage.JS: ([], [], []),  # Used as a placeholder - JS check is done differently
            SupportedLanguage.PHP: (
                ["include_expression", "include_once_expression", "require_expression", "require_once_expression"],
                [], []
            )
        }

        import_nodes = []
        cursor = tree.walk()
        reached_root = False
        while not reached_root:
            node = cursor.node
            if language is SupportedLanguage.JS:
                # JS is too different to fit in the generic format :(
                if self._is_js_import(node):
                    import_nodes.append(node)
            elif len(IMPORT_TYPES_MAP[language][1]) == 0 and len(IMPORT_TYPES_MAP[language][2]) == 0:
                if node.type in IMPORT_TYPES_MAP[language][0]:
                    import_nodes.append(node)
            else:
                if node.type in IMPORT_TYPES_MAP[language][0]:
                    match_idx = IMPORT_TYPES_MAP[language][0].index(node.type)
                    req_parent_type = IMPORT_TYPES_MAP[language][1][match_idx]
                    req_child_types = IMPORT_TYPES_MAP[language][2][match_idx]

                    child_types = [child.type for child in node.children]

                    if req_parent_type == "":
                        parent_type_match = True
                    else:
                        parent_type_match = node.parent.type == req_parent_type
                    if not req_child_types:
                        child_type_match = True
                    else:
                        child_type_match = sorted(child_types) == sorted(req_child_types)

                    if parent_type_match and child_type_match:
                        import_nodes.append(node)
            # Try to goto first child, if no children then goto next sibling
            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue

            # Walk back up when finished on this level
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    reached_root = True
                    break

        return import_nodes

    def _is_js_import(self, node) -> bool:
        if node.type == "import_statement" or node.type == "call_expression":
            if node.type == "call_expression" and node.child(0).text.decode("utf8") == "require":
                return True
            if node.type == "import_statement":
                return True

    def _extract_paths_from_nodes(
        self, nodes: List[Node], repo_path: Path, importer_path: Path, language: SupportedLanguage
    ) -> List[Path]:
        imported_modules = []
        local_imports = []
        project_name = self.target.repo_name.split("-")[-1].lower()
        for node in nodes:
            module_path = None
            if language is SupportedLanguage.PYTHON:
                module_name = node.named_children[0].text.decode("utf8")
                module_path = module_name.replace(".", (str(importer_path.parent)+"/")) + ".py"
            elif language is SupportedLanguage.JAVA:
                target_name = node.child_by_field_name("name")
                if target_name is None:
                    target_name = node.text.decode("utf8").split(" ")[1].strip(";")
                    module_name = target_name
                else:
                    module_name = target_name.text.decode("utf8")
                module_path = module_name.replace('.', '/') + '.java'
            elif language is SupportedLanguage.CPP:
                raw_include = node.text.decode("utf8")
                target = ""
                if '"' in raw_include:
                    # Assume include statement is form: #include "asd/dsa/asd"
                    target = raw_include.split('"')[-2]
                elif "<" in raw_include:
                    # Assume include statement is form: #include <asd/dsa/asd>
                    target = raw_include.split("<")[-1].replace(">","")

                target = target.strip()
                if target.startswith("../../../../../"):
                    module_path = target.replace("../../../../../", (str(importer_path.parents[5])+"/"))
                elif target.startswith("../../../../"):
                    module_path = target.replace("../../../../", (str(importer_path.parents[4])+"/"))
                elif target.startswith("../../../"):
                    module_path = target.replace("../../../", (str(importer_path.parents[3])+"/"))
                elif target.startswith("../../"):
                    module_path = target.replace("../../", (str(importer_path.parents[2])+"/"))
                elif target.startswith("../"):
                    module_path = target.replace("../", (str(importer_path.parents[1])+"/"))
                elif target.startswith("./"):
                    module_path = target.replace("./", (str(importer_path.parent)+"/"))
                else:
                    module_path = target
            elif language is SupportedLanguage.RUBY:
                raw_require = node.text.decode("utf8")
                if raw_require.split(" ")[0] == "require":
                    target = " ".join(raw_require.split(" ")[1:]).replace('"',"").replace("'", "")
                    module_path = target
                elif raw_require.split(" ")[0] == "require_relative":
                    target = " ".join(raw_require.split(" ")[1:]).replace('"', "").replace("'", "")
                    module_path = importer_path.parent.joinpath(target)
                elif raw_require.split(" ")[0] == "include":
                    target = " ".join(raw_require.split(" ")[1:]).replace('"', "").replace("'", "").replace("::","/")
                    module_path = target
            elif language is SupportedLanguage.JS:
                raw_node = node.text.decode("utf8")
                if raw_node.startswith("require"):
                    target = raw_node.split("(")[1].replace(")","").replace('"',"").replace("'", "")
                    if target.startswith("../../"):
                        module_path = target.replace("../../", (str(importer_path.parents[2])+"/"))
                    elif target.startswith("../"):
                        module_path = target.replace("../", (str(importer_path.parents[1])+"/"))
                    elif target.startswith("./"):
                        module_path = target.replace("./", (str(importer_path.parent)+"/"))
                    else:
                        module_path = target
            elif language is SupportedLanguage.PHP:
                raw_target = node.children[1].text.decode("utf8")
                clean_target = raw_target.replace("(", "").replace(")", "").replace("'", "").replace('"',"")
                module_path = clean_target
            if module_path is not None:
                imported_modules.append(module_path)
        if language in [SupportedLanguage.PYTHON, SupportedLanguage.JAVA, SupportedLanguage.RUBY]:
            local_imports += [repo_path.joinpath(mod_path) for mod_path in imported_modules if project_name in str(mod_path)]
        elif language is SupportedLanguage.CPP:
            for mod_path in imported_modules:
                is_file_import = mod_path.endswith(".cpp") or mod_path.endswith(".cc") or mod_path.endswith(".h")
                if is_file_import or '"' in mod_path:
                    local_imports.append(Path(mod_path))
        elif language is SupportedLanguage.JS:
            for mod_path in imported_modules:
                is_file_import = mod_path.endswith(".js")
                if is_file_import or project_name.lower() in str(mod_path).lower():
                    local_imports.append(Path(mod_path))
        elif language is SupportedLanguage.PHP:
            local_imports += [
                repo_path.joinpath(mod_path) for mod_path in imported_modules if not mod_path.startswith("vendor")
            ]
        return local_imports

    def _map_to_known_file(self, repo_path: Path, abs_file_path: Path) -> Optional[str]:
        known_files: List[str] = self.target.known_files
        if str(repo_path) in str(abs_file_path):
            relative_to_repo = abs_file_path.relative_to(repo_path)
        else:
            relative_to_repo = abs_file_path

        if str(relative_to_repo) in known_files:
            return str(relative_to_repo)

        # Maybe import is of a class inside a file
        one_step_up = str(relative_to_repo).replace(f"/{relative_to_repo.name}", relative_to_repo.suffix)
        if str(one_step_up) in known_files:
            return str(one_step_up)

        # Maybe import didn't include the src dir
        relative_to_src = Path("src").joinpath(relative_to_repo)
        if str(relative_to_src) in known_files:
            return str(relative_to_src)

        # Maybe import didn't include the lib dir
        relative_to_lib = Path("lib").joinpath(relative_to_repo)
        if str(relative_to_lib) in known_files:
            return str(relative_to_lib)

        # Maybe project dir is nested inside lib
        project_name = self.target.repo_name.split("-")[-1].lower()
        relative_to_nested_lib = Path(f"lib/{project_name}").joinpath(relative_to_repo)
        if str(relative_to_nested_lib) in known_files:
            return str(relative_to_nested_lib)

        # Maybe import is of a class inside a file and the import didn't include the src dir
        one_step_up_in_src = str(relative_to_src).replace(f"/{relative_to_src.name}", relative_to_src.suffix)
        if str(one_step_up_in_src) in known_files:
            return str(one_step_up_in_src)

        # Try a fuzzy string search as a last resort
        fuzzy_confidences = [fuzz.partial_ratio(str(relative_to_repo), known) for known in known_files]
        most_likely_match = Path(known_files[np.argmax(fuzzy_confidences)])
        if most_likely_match.name.replace(most_likely_match.suffix,"") in str(relative_to_repo):
            logger.debug(f"Fuzzy search for {relative_to_repo} found {most_likely_match}")
            return str(most_likely_match)
        else:
            return None

    def _calculate_file_importance_scores(self, dependency_graph) -> Dict[str, float]:
        importance_metric = self._IMPORTANCE_METRIC_MAP[self.config.file_importance_metric]
        importance_scores = importance_metric(dependency_graph, weight="weight")
        return importance_scores

    def _build_author_map(self):
        """Identify authors for all non-third-party files"""
        author_map: Dict[str, List[str]] = dict()
        for file_path, commit_hashes in self.target.commits_by_file.items():
            if self._is_third_party(file_path):
                continue
            relevant_commits = self.target.get_commits(self.target.commits, commit_hashes)[::-1]
            # Each candidate_authors entry is Tuple: (first contributor flag, num changes made)
            candidate_authors = self._extract_candidate_authors(relevant_commits)
            # Calculate DoA for each candidate author
            abs_doas = self._calc_doa_vals(candidate_authors)
            # Normalize the DoAs between 0 and 1
            abs_doa_max = max(abs_doas.values())
            normed_doas = {author: abs_doa/abs_doa_max for author, abs_doa in abs_doas.items()}
            # Assign File Authorship
            author_map[file_path] = self._find_authors(normed_doas, abs_doas)
        return author_map

    def _is_third_party(self, file_path: str) -> bool:
        """
        Detects using the patterns defined in github-linguist/linguist if the file_path
        points to a third party file or if its actually part of the code base.
        """
        is_third_party = False
        is_third_party = bool(self._third_party_detection_regex.match(file_path))
        return is_third_party

    def _extract_candidate_authors(self, relevant_commits: List[CommitSM]) -> Dict[str, Tuple[bool, float]]:
        candidate_authors: Dict[str, Tuple[bool, float]] = dict()
        for commit in relevant_commits:
            if commit.author_email is not None and "@" in commit.author_email:
                email = commit.author_email
            else:
                email = None
            author = self.target.unique_contributors.get(commit.author_name, email)
            author_key = author.get_key()
            additions = 0
            deletions = 0
            for change_stat in commit.changes:
                additions += change_stat.addition_count
                deletions += change_stat.deletion_count
            if author_key in candidate_authors:
                _scratch = candidate_authors[author_key]
                candidate_authors[author_key] = (_scratch[0], _scratch[1] + additions + deletions)
            else:
                # First author of the file will be whoever wrote the first commit
                is_first_author = relevant_commits.index(commit) == 0
                candidate_authors[author_key] = (is_first_author, (additions + deletions))
        return candidate_authors

    def _calc_doa_vals(self, candidate_authors: Dict[str, Tuple[bool, float]]) -> Dict[str, float]:
        abs_degrees_of_authorship: Dict[str, float] = dict()
        for author in candidate_authors.keys():
            others_change_count = 0
            for key, info in candidate_authors.items():
                if key == author:
                    continue
                others_change_count += info[1]

            doa_abs = self.config.base_doa
            weighted_first_authorship = candidate_authors[author][0] * self.config.first_authorship_weight
            weighted_change_count = candidate_authors[author][1] * self.config.num_changes_weight
            weighted_others_change_count = math.log(1+others_change_count) * self.config.others_change_count_weight
            doa_abs += weighted_first_authorship + weighted_change_count - weighted_others_change_count
            abs_degrees_of_authorship[author] = doa_abs
        return abs_degrees_of_authorship

    def _find_authors(self, norm_doas: Dict[str, float], abs_doas: Dict[str, float]) -> List[str]:
        authors = []
        for author, doa_norm in norm_doas.items():
            if doa_norm > self.config.doa_norm_threshold and abs_doas[author] >= self.config.doa_abs_threshold:
                authors.append(author)
        return authors

    @staticmethod
    def _invert_file_author_map(file_author_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        author_file_map = dict()
        for file_path, authors in file_author_map.items():
            for author in authors:
                if author not in author_file_map:
                    author_file_map[author] = [file_path]
                else:
                    author_file_map[author].append(file_path)
        return author_file_map

    @staticmethod
    def _calc_coverage(file_author_map: Dict[str, List[str]], file_importance: Dict[str, float]) -> float:
        total_importance = sum(file_importance.values())
        orphan_files = [file_path for file_path, authors in file_author_map.items() if len(authors) == 0]

        # If graph analysis fails just fallback on stock avelino method
        if total_importance == 0:
            logger.warning(f"Total File Importance = 0 - Falling Back to Avelino Coverage measure (file count)")
            file_count = len(file_author_map.keys())
            return (file_count - len(orphan_files)) / file_count

        orphan_file_importance = sum([file_importance[file_path] for file_path in orphan_files])
        return (total_importance-orphan_file_importance) / total_importance

    @staticmethod
    def _pop_author_from_map(file_author_map: Dict[str, List[str]], author: str):
        for file_path, authors in file_author_map.items():
            if author in authors:
                authors.remove(author)
