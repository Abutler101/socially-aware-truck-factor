from __future__ import annotations
import re
from typing import Optional, Set, List, Dict

from loguru import logger
from pydantic import BaseModel


class Contributor(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    aliases: Set[str] = set()

    commits_made: Set[str] = set()
    issues_created: Set[int] = set()
    prs_created: Set[int] = set()
    issues_commented_on: Set[int] = set()
    prs_commented_on: Set[int] = set()
    extras: dict = dict()

    def merge(self, other: Contributor) -> Contributor:
        if self == other:
            return self
        return Contributor(
            name=self.name or other.name,
            email=self.email or other.email,
            aliases=self.aliases.union(other.aliases),
            commits_made=self.commits_made.union(other.commits_made),
            issues_created=self.issues_created.union(other.issues_created),
            prs_created=self.prs_created.union(other.prs_created),
            issues_commented_on=self.issues_commented_on.union(other.issues_commented_on),
            prs_commented_on=self.prs_commented_on.union(other.prs_commented_on),
            extras={**self.extras, **other.extras},
        )

    def get_key(self) -> str:
        if self.name is not None and self.email is not None:
            return f"{self.name.lower()}-{self.email.lower()}"
        elif self.name is not None and self.email is None:
            return f"{self.name.lower()}-UNKNOWN"
        elif self.name is None and self.email is not None:
            return f"UNKNOWN-{self.email.lower()}"


class UniqueContributorStore(BaseModel):
    # Key is Name-Email, if one of the values isn't known, placeholder value UNKNOWN is used
    store: Dict[str, Contributor] = dict()

    def add_or_update(self, contributor: Contributor):
        if contributor.email is not None and "@" not in contributor.email:
            contributor.email = None
        existing_key = self._find_match(contributor)
        if existing_key is None:
            self.store[contributor.get_key()] = contributor
        else:
            partial = self.store[existing_key]
            updated = partial.merge(contributor)
            updated.aliases.add(contributor.get_key())
            self.store.pop(existing_key)
            self.store[updated.get_key()] = updated

    def drop(self, contributor: Contributor):
        possible_match = self._find_match(contributor)
        self.store.pop(possible_match)

    def get(self, name: Optional[str] = None, email: Optional[str] = None) -> Optional[Contributor]:
        if name is None and email is None:
            return None
        possible_match = self._find_match(Contributor(name=name, email=email))
        if possible_match is None:
            return None
        return self.store[possible_match]

    def _find_match(self, contributor: Contributor) -> Optional[str]:
        if contributor.name is None:
            target_name = "UNKNOWN"
        else:
            target_name = contributor.name.lower()
            target_name = target_name.replace("?", "\?").replace("*", "\*").replace("+", "\+")
        if contributor.email is None:
            target_email = "UNKNOWN"
        else:
            target_email = contributor.email.lower()

        # Name-Email match
        if f"{target_name}-{target_email}" in self.store:
            return f"{target_name}-{target_email}"

        # If no match Email only match
        if target_email != "UNKNOWN":
            try:
                re_search = re.compile(f".*-{target_email}")
            except re.error as err:
                logger.warning(f"Regex Search Broken by Contributor {contributor} email address")
                return None
            possible_matches = list(filter(re_search.match, self.store.keys()))
            if len(possible_matches) > 0:
                return possible_matches[0]

        # If no match Name only match
        if target_name != "UNKNOWN":
            re_search = re.compile(f"{target_name}-.*")
            possible_matches = list(filter(re_search.match, self.store.keys()))
            if len(possible_matches) > 0:
                return possible_matches[0]

        # If no match Name only with whitespace allowance match
        if target_name != "UNKNOWN":
            split_on_spaces = target_name.split(" ")
            with_optional_whitespace = "\s*".join(split_on_spaces)
            re_search = re.compile(f"{with_optional_whitespace}-.*")
            possible_matches = list(filter(re_search.match, self.store.keys()))
            if len(possible_matches) > 0:
                return possible_matches[0]

        # If no match Email user only match
        if target_email != "UNKNOWN":
            try:
                re_search = re.compile(f".*-{target_email.split('@')[0]}.*")
            except re.error as err:
                logger.warning(f"Regex Search Broken by Contributor {contributor} email address")
                return None

            possible_matches = list(filter(re_search.match, self.store.keys()))
            if len(possible_matches) > 0:
                return possible_matches[0]

        # If no match - no match present
        return None
