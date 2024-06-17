#!/usr/bin/env python3

# NB: the following functions are used in Meta-internal workflows
# (github_first_try_merge/my_handler.py) and thus have functionality limitations
# (no `git` command access, no network access besides the strict allow list):
#
# find_matching_merge_rule
# read_merge_rules
#
# Also any signature changes of these functions, as well as changes to the `GitHubPR`
# class, will likely require corresponding changes for the internal workflows.

import os
import re
from functools import lru_cache
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)
from warnings import warn

from github_utils import (
    gh_fetch_merge_base,
    gh_graphql,
    gh_post_pr_comment,
    GitHubComment,
)

from gitutils import (
    are_ghstack_branches_in_sync,
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
)


class JobCheckState(NamedTuple):
    name: str
    url: str
    status: Optional[str]


JobNameToStateDict = Dict[str, JobCheckState]


class WorkflowCheckState:
    def __init__(self, name: str, url: str, status: Optional[str]):
        self.name: str = name
        self.url: str = url
        self.status: Optional[str] = status
        self.jobs: JobNameToStateDict = {}


GH_PR_REVIEWS_FRAGMENT = """
fragment PRReviews on PullRequestReviewConnection {
  nodes {
    author {
      login
    }
    bodyText
    createdAt
    authorAssociation
    editor {
      login
    }
    databaseId
    url
    state
  }
  pageInfo {
    startCursor
    hasPreviousPage
  }
}
"""

GH_CHECKSUITES_FRAGMENT = """
fragment PRCheckSuites on CheckSuiteConnection {
  edges {
    node {
      app {
        name
        databaseId
      }
      workflowRun {
        workflow {
          name
        }
        databaseId
        url
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
          databaseId
          title
          summary
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      conclusion
    }
    cursor
  }
  pageInfo {
    hasNextPage
  }
}
"""

GH_COMMIT_AUTHORS_FRAGMENT = """
fragment CommitAuthors on PullRequestCommitConnection {
  nodes {
    commit {
      authors(first: 2) {
        nodes {
          user {
            login
          }
          email
          name
        }
      }
      oid
    }
  }
  pageInfo {
    endCursor
    hasNextPage
  }
}
"""

GH_GET_PR_INFO_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + GH_CHECKSUITES_FRAGMENT
    + GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closed
      author {
        login
      }
      title
      body
      headRefName
      headRepository {
        nameWithOwner
      }
      baseRefName
      baseRefOid
      baseRepository {
        nameWithOwner
        isPrivate
        defaultBranchRef {
          name
        }
      }
      commits_with_authors: commits(first: 100) {
        ...CommitAuthors
        totalCount
      }
      commits(last: 1) {
        nodes {
          commit {
            checkSuites(first: 10) {
              ...PRCheckSuites
            }
            status {
              contexts {
                context
                state
                targetUrl
              }
            }
            oid
          }
        }
      }
      reviews(last: 100) {
        ...PRReviews
      }
      comments(last: 5) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
    }
  }
}
"""
)


GH_GET_PR_NEXT_CHECKSUITES = (
    GH_CHECKSUITES_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 10, after: $cursor) {
              ...PRCheckSuites
            }
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $number: Int!, $cs_cursor: String, $cr_cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 1, after: $cs_cursor) {
              nodes {
                checkRuns(first: 100, after: $cr_cursor) {
                  nodes {
                    name
                    conclusion
                    detailsUrl
                    databaseId
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_PREV_COMMENTS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      comments(last: 100, before: $cursor) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
    }
  }
}
"""

# This query needs read-org permission
GH_GET_TEAM_MEMBERS_QUERY = """
query($org: String!, $name: String!, $cursor: String) {
  organization(login: $org) {
    team(slug: $name) {
      members(first: 100, after: $cursor) {
        nodes {
          login
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_AUTHORS_QUERY = (
    GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits_with_authors: commits(first: 100, after: $cursor) {
        ...CommitAuthors
      }
    }
  }
}
"""
)

GH_GET_PR_PREV_REVIEWS_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      reviews(last: 100, before: $cursor) {
        ...PRReviews
      }
    }
  }
}
"""
)

GH_GET_REPO_SUBMODULES = """
query ($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    submodules(first: 100) {
      nodes {
        path
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_DESC = re.compile(r"Stack.*:\r?\n(\* [^\r\n]+\r?\n)+", re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r"Pull Request resolved: "
    r"https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)",
    re.MULTILINE,
)
RE_PR_CC_LINE = re.compile(r"^cc:? @\w+.*\r?\n?$", re.MULTILINE)
RE_DIFF_REV = re.compile(r"^Differential Revision:.+?(D[0-9]+)", re.MULTILINE)
REMOTE_MAIN_BRANCH = "origin/main"
INTERNAL_CHANGES_CHECKRUN_NAME = "Meta Internal-Only Changes Check"


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


@lru_cache(maxsize=None)
def gh_get_team_members(org: str, name: str) -> List[str]:
    rc: List[str] = []
    team_members: Dict[str, Any] = {
        "pageInfo": {"hasNextPage": "true", "endCursor": None}
    }
    while bool(team_members["pageInfo"]["hasNextPage"]):
        query = gh_graphql(
            GH_GET_TEAM_MEMBERS_QUERY,
            org=org,
            name=name,
            cursor=team_members["pageInfo"]["endCursor"],
        )
        team = query["data"]["organization"]["team"]
        if team is None:
            warn(f"Requested non-existing team {org}/{name}")
            return []
        team_members = team["members"]
        rc += [member["login"] for member in team_members["nodes"]]
    return rc


def get_check_run_name_prefix(workflow_run: Any) -> str:
    if workflow_run is None:
        return ""
    else:
        return f'{workflow_run["workflow"]["name"]} / '


def is_passing_status(status: Optional[str]) -> bool:
    return status is not None and status.upper() in ["SUCCESS", "SKIPPED", "NEUTRAL"]


def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[List[Dict[str, Dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any],
) -> JobNameToStateDict:
    # graphql seems to favor the most recent workflow run, so in theory we
    # shouldn't need to account for reruns, but do it just in case

    # workflow -> job -> job info
    workflows: Dict[str, WorkflowCheckState] = {}

    # for the jobs that don't have a workflow
    no_workflow_obj: WorkflowCheckState = WorkflowCheckState("", "", None)

    def add_conclusions(edges: Any) -> None:
        for edge_idx, edge in enumerate(edges):
            node = edge["node"]
            workflow_run = node["workflowRun"]
            checkruns = node["checkRuns"]

            workflow_obj: WorkflowCheckState = no_workflow_obj

            if workflow_run is not None:
                workflow_name = workflow_run["workflow"]["name"]
                workflow_conclusion = node["conclusion"]
                # Do not override existing status with cancelled
                if workflow_conclusion == "CANCELLED" and workflow_name in workflows:
                    continue
                if workflow_name not in workflows:
                    workflows[workflow_name] = WorkflowCheckState(
                        name=workflow_name,
                        status=workflow_conclusion,
                        url=workflow_run["url"],
                    )
                workflow_obj = workflows[workflow_name]

            while checkruns is not None:
                for checkrun_node in checkruns["nodes"]:
                    if not isinstance(checkrun_node, dict):
                        warn(f"Expected dictionary, but got {type(checkrun_node)}")
                        continue
                    checkrun_name = f'{get_check_run_name_prefix(workflow_run)}{checkrun_node["name"]}'
                    existing_checkrun = workflow_obj.jobs.get(checkrun_name)
                    if existing_checkrun is None or not is_passing_status(
                        existing_checkrun.status
                    ):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(
                            checkrun_name,
                            checkrun_node["detailsUrl"],
                            checkrun_node["conclusion"],
                        )

                if bool(checkruns["pageInfo"]["hasNextPage"]):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None

    all_edges = checksuites["edges"].copy()
    while bool(checksuites["pageInfo"]["hasNextPage"]):
        checksuites = get_next_checksuites(checksuites)
        all_edges.extend(checksuites["edges"])

    add_conclusions(all_edges)

    # Flatten the dictionaries.  If there exists jobs in the workflow run, put
    # the jobs in but don't put the workflow in.  We care more about the jobs in
    # the workflow that ran than the container workflow.
    res: JobNameToStateDict = {}
    for workflow_name, workflow in workflows.items():
        if len(workflow.jobs) > 0:
            for job_name, job in workflow.jobs.items():
                res[job_name] = job
        else:
            res[workflow_name] = JobCheckState(
                workflow.name,
                workflow.url,
                workflow.status,
            )
    for job_name, job in no_workflow_obj.jobs.items():
        res[job_name] = job
    return res


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def can_skip_internal_checks(pr: "GitHubPR", comment_id: Optional[int] = None) -> bool:
    if comment_id is None:
        return False
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return False
    return comment.author_login == "facebook-github-bot"


def _revlist_to_prs(
    repo: GitRepo,
    pr: "GitHubPR",
    rev_list: Iterable[str],
    should_skip: Optional[Callable[[int, "GitHubPR"], bool]] = None,
) -> List[Tuple["GitHubPR", str]]:
    rc: List[Tuple[GitHubPR, str]] = []
    for idx, rev in enumerate(rev_list):
        msg = repo.commit_message(rev)
        m = RE_PULL_REQUEST_RESOLVED.search(msg)
        if m is None:
            raise RuntimeError(
                f"Could not find PR-resolved string in {msg} of ghstacked PR {pr.pr_num}"
            )
        if pr.org != m.group("owner") or pr.project != m.group("repo"):
            raise RuntimeError(
                f"PR {m.group('number')} resolved to wrong owner/repo pair"
            )
        pr_num = int(m.group("number"))
        candidate = GitHubPR(pr.org, pr.project, pr_num) if pr_num != pr.pr_num else pr
        if should_skip is not None and should_skip(idx, candidate):
            continue
        rc.append((candidate, rev))
    return rc


def get_ghstack_prs(
    repo: GitRepo, pr: "GitHubPR", open_only: bool = True
) -> List[Tuple["GitHubPR", str]]:
    """
    Get the PRs in the stack that are below this PR (inclusive).  Throws error if any of the open PRs are out of sync.
    @:param open_only: Only return open PRs
    """
    # For ghstack, cherry-pick commits based from origin
    orig_ref = f"{repo.remote}/{pr.get_ghstack_orig_ref()}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")

    def skip_func(idx: int, candidate: "GitHubPR") -> bool:
        if not open_only or not candidate.is_closed():
            return False
        print(
            f"Skipping {idx+1} of {len(rev_list)} PR (#{candidate.pr_num}) as its already been merged"
        )
        return True

    assert pr.is_ghstack_pr()
    entire_stack = _revlist_to_prs(repo, pr, reversed(rev_list), skip_func)

    for stacked_pr, rev in entire_stack:
        if stacked_pr.is_closed():
            continue
        base_ref = stacked_pr.base_ref()
        if base_ref == pr.default_branch():
            base_ref = repo.get_merge_base(
                f"{repo.remote}/{base_ref}", f"{repo.remote}/{stacked_pr.head_ref()}"
            )
        if not are_ghstack_branches_in_sync(repo, stacked_pr.head_ref(), base_ref):
            raise RuntimeError(
                f"PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on "
                + f"branch {stacked_pr.get_ghstack_orig_ref()} that would be merged into {stacked_pr.default_branch()}.  "
                + "This usually happens because there is a non ghstack change in the PR.  "
                + f"Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack)."
            )
    return entire_stack


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.labels: Optional[List[str]] = None
        self.conclusions: Optional[JobNameToStateDict] = None
        self.comments: Optional[List[GitHubComment]] = None
        self._authors: Optional[List[Tuple[str, str]]] = None
        self._reviews: Optional[List[Tuple[str, str]]] = None
        self.merge_base: Optional[str] = None
        self.submodules: Optional[List[str]] = None

    def is_closed(self) -> bool:
        return bool(self.info["closed"])

    def base_ref(self) -> str:
        return cast(str, self.info["baseRefName"])

    def default_branch(self) -> str:
        return cast(str, self.info["baseRepository"]["defaultBranchRef"]["name"])

    def head_ref(self) -> str:
        return cast(str, self.info["headRefName"])

    def is_ghstack_pr(self) -> bool:
        return RE_GHSTACK_HEAD_REF.match(self.head_ref()) is not None

    def get_ghstack_orig_ref(self) -> str:
        assert self.is_ghstack_pr()
        return re.sub(r"/head$", "/orig", self.head_ref())

    def last_commit(self) -> Any:
        return self.info["commits"]["nodes"][-1]["commit"]

    def get_merge_base(self) -> str:
        if self.merge_base:
            return self.merge_base

        last_commit_oid = self.last_commit()["oid"]
        # NB: We could use self.base_ref() here for regular PR, however, that doesn't
        # work for ghstack where the base is the custom branch, i.e. gh/USER/ID/base,
        # so let's just use main instead
        self.merge_base = gh_fetch_merge_base(
            self.org, self.project, last_commit_oid, self.default_branch()
        )

        # Fallback to baseRefOid if the API call fails, i.e. rate limit. Note that baseRefOid
        # points to the base ref associated with the PR or, in other words, the head of main
        # when the PR is created or rebased. This is not necessarily the merge base commit,
        # but it could serve as a fallback in most cases and it's readily available as part
        # of the PR info
        if not self.merge_base:
            self.merge_base = cast(str, self.info["baseRefOid"])

        return self.merge_base

    def _get_reviews(self) -> List[Tuple[str, str]]:
        if self._reviews is None:
            self._reviews = []
            info = self.info
            for _ in range(100):
                nodes = info["reviews"]["nodes"]
                self._reviews = [
                    (node["author"]["login"], node["state"]) for node in nodes
                ] + self._reviews
                if not info["reviews"]["pageInfo"]["hasPreviousPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_PREV_REVIEWS_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["reviews"]["pageInfo"]["startCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
        reviews = {}
        for author, state in self._reviews:
            if state != "COMMENTED":
                reviews[author] = state
        return list(reviews.items())

    def get_approved_by(self) -> List[str]:
        return [login for (login, state) in self._get_reviews() if state == "APPROVED"]

    def get_pr_creator_login(self) -> str:
        return cast(str, self.info["author"]["login"])

    def _fetch_authors(self) -> List[Tuple[str, str]]:
        if self._authors is not None:
            return self._authors
        authors: List[Tuple[str, str]] = []

        def add_authors(info: Dict[str, Any]) -> None:
            for node in info["commits_with_authors"]["nodes"]:
                for author_node in node["commit"]["authors"]["nodes"]:
                    user_node = author_node["user"]
                    author = f"{author_node['name']} <{author_node['email']}>"
                    if user_node is None:
                        # If author is not github user, user node will be null
                        authors.append(("", author))
                    else:
                        authors.append((cast(str, user_node["login"]), author))

        info = self.info
        for _ in range(100):
            add_authors(info)
            if not info["commits_with_authors"]["pageInfo"]["hasNextPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_NEXT_AUTHORS_QUERY,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["commits_with_authors"]["pageInfo"]["endCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
        self._authors = authors
        return authors

    def get_committer_login(self, num: int = 0) -> str:
        return self._fetch_authors()[num][0]

    def get_committer_author(self, num: int = 0) -> str:
        return self._fetch_authors()[num][1]

    def get_checkrun_conclusions(self) -> JobNameToStateDict:
        """Returns dict of checkrun -> [conclusion, url]"""
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.last_commit()

        def get_pr_next_check_runs(
            edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any
        ) -> Any:
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECK_RUNS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                cr_cursor=checkruns["pageInfo"]["endCursor"],
            )
            last_commit = rc["data"]["repository"]["pullRequest"]["commits"]["nodes"][
                -1
            ]["commit"]
            checkruns = last_commit["checkSuites"]["nodes"][-1]["checkRuns"]
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECKSUITES,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=checksuites["edges"][-1]["cursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
            last_commit = info["commits"]["nodes"][-1]["commit"]
            if last_commit["oid"] != orig_last_commit["oid"]:
                raise RuntimeError("Last commit changed on PR")
            return last_commit["checkSuites"]

        checksuites = orig_last_commit["checkSuites"]

        self.conclusions = add_workflow_conclusions(
            checksuites, get_pr_next_check_runs, get_pr_next_checksuites
        )

        # Append old style statuses(like ones populated by CircleCI or EasyCLA) to conclusions
        if orig_last_commit["status"] and orig_last_commit["status"]["contexts"]:
            for status in orig_last_commit["status"]["contexts"]:
                name = status["context"]
                self.conclusions[name] = JobCheckState(
                    name,
                    status["targetUrl"],
                    status["state"],
                )

        return self.conclusions

    def get_authors(self) -> Dict[str, str]:
        rc = {}
        for idx in range(len(self._fetch_authors())):
            rc[self.get_committer_login(idx)] = self.get_committer_author(idx)

        return rc

    def get_author(self) -> str:
        authors = self.get_authors()
        if len(authors) == 1:
            return next(iter(authors.values()))
        creator = self.get_pr_creator_login()
        # If PR creator is not among authors
        # Assume it was authored by first commit author
        if creator not in authors:
            return self.get_committer_author(0)
        return authors[creator]

    def get_title(self) -> str:
        return cast(str, self.info["title"])

    def get_body(self) -> str:
        return cast(str, self.info["body"])

    def get_pr_url(self) -> str:
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    @staticmethod
    def _comment_from_node(node: Any) -> GitHubComment:
        editor = node["editor"]
        return GitHubComment(
            body_text=node["bodyText"],
            created_at=node["createdAt"] if "createdAt" in node else "",
            author_login=node["author"]["login"],
            author_association=node["authorAssociation"],
            editor_login=editor["login"] if editor else None,
            database_id=node["databaseId"],
            url=node["url"],
        )

    def get_comments(self) -> List[GitHubComment]:
        if self.comments is not None:
            return self.comments
        self.comments = []
        info = self.info["comments"]
        # Do not try to fetch more than 10K comments
        for _ in range(100):
            self.comments = [
                self._comment_from_node(node) for node in info["nodes"]
            ] + self.comments
            if not info["pageInfo"]["hasPreviousPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_PREV_COMMENTS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["pageInfo"]["startCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]["comments"]
        return self.comments

    def get_last_comment(self) -> GitHubComment:
        return self._comment_from_node(self.info["comments"]["nodes"][-1])

    def get_comment_by_id(self, database_id: int) -> GitHubComment:
        if self.comments is None:
            # Fastpath - try searching in partial prefetched comments
            for node in self.info["comments"]["nodes"]:
                comment = self._comment_from_node(node)
                if comment.database_id == database_id:
                    return comment

        for comment in self.get_comments():
            if comment.database_id == database_id:
                return comment

        # The comment could have actually been a review left on the PR (the message written alongside the review).
        # (This is generally done to trigger the merge right when a comment is left)
        # Check those review comments to see if one of those was the comment in question.
        for node in self.info["reviews"]["nodes"]:
            # These review comments contain all the fields regular comments need
            comment = self._comment_from_node(node)
            if comment.database_id == database_id:
                return comment

        raise RuntimeError(f"Comment with id {database_id} not found")

    def get_diff_revision(self) -> Optional[str]:
        rc = RE_DIFF_REV.search(self.get_body())
        return rc.group(1) if rc is not None else None

    def has_internal_changes(self) -> bool:
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        if self.get_diff_revision() is None:
            return False
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].status != "SUCCESS"

    def merge_ghstack_into(
        self,
        repo: GitRepo,
        comment_id: Optional[int] = None,
    ) -> List["GitHubPR"]:
        assert self.is_ghstack_pr()
        ghstack_prs = get_ghstack_prs(
            repo, self, open_only=False
        )  # raises error if out of sync
        pr_dependencies = []
        for pr, rev in ghstack_prs:
            if pr.is_closed():
                pr_dependencies.append(pr)
                continue

            commit_msg = pr.gen_commit_message(
                filter_ghstack=True, ghstack_deps=pr_dependencies
            )
            if pr.pr_num != self.pr_num:
                # Raises exception if matching rule is not found
                find_matching_merge_rule(
                    pr,
                    comment_id=comment_id,
                )
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
            pr_dependencies.append(pr)
        return [x for x, _ in ghstack_prs if not x.is_closed()]

    def gen_commit_message(
        self,
        filter_ghstack: bool = False,
        ghstack_deps: Optional[List["GitHubPR"]] = None,
    ) -> str:
        """Fetches title and body from PR description
        adds reviewed by, pull request resolved and optionally
        filters out ghstack info"""
        # Adding the url here makes it clickable within the Github UI
        approved_by_urls = ", ".join(
            prefix_with_github_url(login) for login in self.get_approved_by()
        )
        # Remove "cc: " line from the message body
        msg_body = re.sub(RE_PR_CC_LINE, "", self.get_body())
        if filter_ghstack:
            msg_body = re.sub(RE_GHSTACK_DESC, "", msg_body)
        msg = self.get_title() + f" (#{self.pr_num})\n\n"
        msg += msg_body

        # Mention PR co-authors
        for author_login, author_name in self.get_authors().items():
            if author_login != self.get_pr_creator_login():
                msg += f"\nCo-authored-by: {author_name}"

        msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
        msg += f"Approved by: {approved_by_urls}\n"
        if ghstack_deps:
            msg += f"ghstack dependencies: {', '.join([f'#{pr.pr_num}' for pr in ghstack_deps])}\n"
        return msg

    def merge_into(
        self,
        repo: GitRepo,
        *,
        dry_run: bool = False,
        comment_id: Optional[int] = None,
    ) -> None:
        # Raises exception if matching rule is not found
        find_matching_merge_rule(
            self,
            comment_id=comment_id,
        )
        repo.push(self.default_branch(), dry_run)

    def merge_changes(
        self,
        repo: GitRepo,
        comment_id: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> List["GitHubPR"]:
        """
        :param skip_all_rule_checks: If true, skips all rule checks, useful for dry-running merge locally
        """
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)
        if not self.is_ghstack_pr():
            msg = self.gen_commit_message()
            pr_branch_name = f"__pull-request-{self.pr_num}__init__"
            repo.fetch(f"pull/{self.pr_num}/head", pr_branch_name)
            repo._run_git("merge", "--squash", pr_branch_name)
            repo._run_git("commit", f'--author="{self.get_author()}"', "-m", msg)
            return []
        else:
            return self.merge_ghstack_into(
                repo,
                comment_id=comment_id,
            )


class MergeRuleFailedError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def find_matching_merge_rule(
    pr: GitHubPR,
    comment_id: Optional[int] = None,
) -> None:
    """
    Returns merge rule matching to this pr together with the list of associated pending
    and failing jobs OR raises an exception.

    NB: this function is used in Meta-internal workflows, see the comment at the top of
    this file for details.
    """
    if pr.has_internal_changes() and not can_skip_internal_checks(
        pr, comment_id=comment_id
    ):
        raise MergeRuleFailedError(
            f"PR {pr.pr_num} has internal changes and must be merged internally"
        )

    checks = pr.get_checkrun_conclusions()
    if "Facebook CLA Check" not in checks or checks["Facebook CLA Check"].status != "SUCCESS":
        raise MergeRuleFailedError(f"PR {pr.pr_num} does not have a signed CLA")

    approved_by = set(pr.get_approved_by())
    approvers = gh_get_team_members("pytorch", "metamates")
    if any(approver in approvers for approver in approved_by):
        return
    else:
        raise MergeRuleFailedError(f"PR {pr.pr_num} is not approved by a metamate")


def prefix_with_github_url(suffix_str: str) -> str:
    return f"https://github.com/{suffix_str}"


def merge(
    pr: GitHubPR,
    repo: GitRepo,
    comment_id: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    initial_commit_sha = pr.last_commit()["oid"]
    pr_link = f"https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num}"
    print(f"Attempting merge of {initial_commit_sha} ({pr_link})")

    if pr.is_ghstack_pr():
        get_ghstack_prs(repo, pr)  # raises error if out of sync

    return pr.merge_into(
        repo,
        dry_run=dry_run,
        comment_id=comment_id,
    )


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, title: str = "Merge failed") -> None:
        exception = f"**Reason**: {e}"

        internal_debugging = ""
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            # Hide this behind a collapsed bullet since it's not helpful to most devs
            internal_debugging = "\n".join(
                line
                for line in (
                    "<details><summary>Details for Dev Infra team</summary>",
                    f'Raised by <a href="{run_url}">workflow job</a>\n',
                    "</details>",
                )
                if line
            )  # ignore empty lines during the join

        msg = "\n".join((f"## {title}", f"{exception}", "", f"{internal_debugging}"))

        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback

        traceback.print_exc()

    # if pr.is_closed():
    #     gh_post_pr_comment(
    #         org,
    #         project,
    #         args.pr_num,
    #         f"Can't merge closed PR #{args.pr_num}",
    #         dry_run=args.dry_run,
    #     )
    #     return

    try:
        merge(
            pr,
            repo,
            dry_run=args.dry_run,
            comment_id=args.comment_id,
        )
    except Exception as e:
        handle_exception(e)


if __name__ == "__main__":
    main()
