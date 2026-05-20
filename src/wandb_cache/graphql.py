from __future__ import annotations

import base64
import json
import os
import random
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from wandb_cache.json import to_jsonable

RUNS_METADATA_QUERY = """
query RunsMetadata($entity: String!, $project: String!, $filters: JSONString, $first: Int!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, first: $first, after: $cursor) {
      edges {
        node {
          name
          displayName
          state
          group
          tags
          createdAt
          config
        }
        cursor
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""


RUNS_METADATA_WITH_SUMMARY_QUERY = """
query RunsMetadata($entity: String!, $project: String!, $filters: JSONString, $first: Int!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, first: $first, after: $cursor) {
      edges {
        node {
          name
          displayName
          state
          group
          tags
          createdAt
          config
          summaryMetrics
        }
        cursor
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""


def fetch_run_metadata_graphql(
    project: str,
    filters: dict[str, Any] | None = None,
    include_summary: bool = False,
    per_page: int = 500,
) -> list[dict[str, Any]]:
    if per_page < 1:
        raise ValueError("per_page must be >= 1")

    entity, project_name = split_project_path(project)
    records = []
    cursor = None
    while True:
        data = execute_graphql(
            query=RUNS_METADATA_WITH_SUMMARY_QUERY if include_summary else RUNS_METADATA_QUERY,
            variables={
                "entity": entity,
                "project": project_name,
                "filters": json.dumps(filters or {}),
                "first": per_page,
                "cursor": cursor,
            },
        )
        project_data = data["data"]["project"]
        if project_data is None:
            raise ValueError(f"Could not find W&B project {project}")

        runs_data = project_data["runs"]
        for edge in runs_data["edges"]:
            records.append(normalize_run_node(edge["node"], include_summary=include_summary))

        page_info = runs_data["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
    return records


def split_project_path(project: str) -> tuple[str, str]:
    parts = project.split("/")
    if len(parts) != 2:
        raise ValueError("project must be in 'entity/project' form")
    return parts[0], parts[1]


def normalize_run_node(node: dict[str, Any], include_summary: bool) -> dict[str, Any]:
    record = {
        "run_id": to_jsonable(node["name"]),
        "run_name": to_jsonable(node["displayName"]),
        "run_state": to_jsonable(node["state"]),
        "run_group": to_jsonable(node["group"]),
        "run_tags": to_jsonable(list(node["tags"] or [])),
        "run_created_at": to_jsonable(node["createdAt"]),
        "config": to_jsonable(decode_run_config(node.get("config"))),
    }
    if include_summary:
        record["summary"] = to_jsonable(decode_json_dict(node.get("summaryMetrics")))
    return record


def decode_run_config(config: object) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, str):
        config = json.loads(config)

    decoded = {}
    for key, value in dict(config).items():
        if key == "_wandb":
            continue
        if isinstance(value, dict) and "value" in value:
            decoded[key] = value["value"]
        else:
            decoded[key] = value
    return decoded


def decode_json_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value)


def execute_graphql(
    query: str,
    variables: dict[str, Any],
    max_retries: int = 6,
    max_retry_delay: float = 30.0,
) -> dict[str, Any]:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    api_key = load_api_key(base_url)

    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    basic_auth = base64.b64encode(f"api:{api_key}".encode("utf-8")).decode("ascii")
    request = Request(
        f"{base_url}/graphql",
        data=payload,
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    data = None
    last_error = None
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as error:
            last_error = error
            if error.code != 429 or attempt == max_retries - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            if retry_after is not None:
                delay = float(retry_after)
            else:
                delay = min(max_retry_delay, (2**attempt) + random.uniform(0.0, 1.0))
            time.sleep(delay)

    if data is None:
        raise last_error

    if data.get("errors"):
        messages = ", ".join(error.get("message", "unknown error") for error in data["errors"])
        raise RuntimeError(messages)
    return data


def load_api_key(base_url: str) -> str:
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return api_key

    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        raise RuntimeError("No WANDB_API_KEY set and ~/.netrc not found")

    import netrc

    machine = base_url.replace("https://", "").replace("http://", "")
    auth_data = netrc.netrc(netrc_path).authenticators(machine)
    if auth_data is None:
        auth_data = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    if auth_data is None or not auth_data[2]:
        raise RuntimeError("No usable W&B API key found in WANDB_API_KEY or ~/.netrc")
    return auth_data[2]
