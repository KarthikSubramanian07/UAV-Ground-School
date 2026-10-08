"""ROS 2 interface definitions (.msg, .srv, .action) and their type hashes.

Parses the ``.msg`` format the way ``rosidl_adapter`` does (fields, defaults,
constants, bounded strings, fixed arrays, bounded and unbounded sequences),
expands services and actions into the messages rosidl generates for them
(``_Request``, ``_Response``, ``_Event``, ``_SendGoal``, ``_GetResult``,
``_FeedbackMessage``), and computes REP 2011 type description hashes
(``RIHS01_...``), the ones ROS 2 Jazzy writes into discovery and checks with
``ros2 topic info --verbose``.

The tests hold this module to every interface installed with ROS 2 Jazzy:
the parsed fields must match ``ros2 interface show`` and every hash must match
the ``.json`` file rosidl generated next to the definition.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Primitive .msg types and their size in bytes in CDR.
PRIMITIVES = {
    "bool": 1,
    "byte": 1,
    "char": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float32": 4,
    "float64": 8,
}
STRINGS = ("string", "wstring")

# REP 2011 / type_description_interfaces/msg/FieldType ids. In .msg files
# ``char`` is an alias of uint8 and ``byte`` is an IDL octet.
_FIELD_TYPE_IDS = {
    "nested": 1,
    "int8": 2,
    "uint8": 3,
    "char": 3,
    "int16": 4,
    "uint16": 5,
    "int32": 6,
    "uint32": 7,
    "int64": 8,
    "uint64": 9,
    "float32": 10,
    "float64": 11,
    "bool": 15,
    "byte": 16,
    "string": 17,
    "wstring": 18,
    "bounded_string": 21,
    "bounded_wstring": 22,
}
_ARRAY_OFFSET = {"": 0, "array": 48, "bounded": 96, "sequence": 144}

_TYPE_RE = re.compile(r"^(?P<base>[A-Za-z][\w/]*)(?:<=(?P<sbound>\d+))?(?:\[(?P<le><=)?(?P<n>\d*)\])?$")
_SRV_PART = re.compile(r"_(Request|Response|Event)$")
_ACTION_PART = re.compile(r"_(Goal|Result|Feedback|FeedbackMessage|SendGoal|GetResult)(_Request|_Response|_Event)?$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class Type:
    """A field type: a primitive, a (bounded) string, or a nested message, maybe in an array."""

    base: str  # primitive name, "string", "wstring", or a full name like "std_msgs/msg/Header"
    string_bound: int = 0
    array: str = ""  # "" scalar, "array" fixed size, "bounded" sequence, "sequence" unbounded
    size: int = 0  # array length or sequence bound

    @property
    def is_primitive(self) -> bool:
        return self.base in PRIMITIVES

    @property
    def is_string(self) -> bool:
        return self.base in STRINGS

    @property
    def is_nested(self) -> bool:
        return not (self.is_primitive or self.is_string)

    def element(self) -> Type:
        return Type(self.base, self.string_bound)

    def __str__(self) -> str:
        text = self.base
        if self.is_nested:
            pkg, _, name = self.base.split("/")
            text = f"{pkg}/{name}"
        if self.string_bound:
            text += f"<={self.string_bound}"
        if self.array == "array":
            text += f"[{self.size}]"
        elif self.array == "bounded":
            text += f"[<={self.size}]"
        elif self.array == "sequence":
            text += "[]"
        return text

    def type_id(self) -> int:
        if self.is_nested:
            key = "nested"
        elif self.is_string and self.string_bound:
            key = "bounded_" + self.base
        else:
            key = self.base
        return _FIELD_TYPE_IDS[key] + _ARRAY_OFFSET[self.array]


@dataclass(frozen=True)
class Field:
    name: str
    type: Type
    default: str | None = None  # the literal text from the .msg file


@dataclass(frozen=True)
class Constant:
    name: str
    type: Type
    value: str


@dataclass
class Message:
    name: str  # e.g. "std_msgs/msg/String" or "example_interfaces/srv/AddTwoInts_Request"
    fields: list[Field] = field(default_factory=list)
    constants: list[Constant] = field(default_factory=list)

    @property
    def package(self) -> str:
        return self.name.split("/")[0]

    @property
    def short(self) -> str:
        return self.name.split("/")[-1]


@dataclass
class Service:
    name: str
    request: Message
    response: Message
    event: Message


@dataclass
class Action:
    name: str
    goal: Message
    result: Message
    feedback: Message
    send_goal: Service
    get_result: Service
    feedback_message: Message


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, but not a ``#`` inside a quoted default."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def parse_type(text: str, package: str) -> Type:
    m = _TYPE_RE.match(text)
    if not m:
        raise ValueError(f"bad type {text!r}")
    base = m.group("base")
    sbound = int(m.group("sbound") or 0)
    if sbound and base not in STRINGS:
        raise ValueError(f"only strings can be bounded: {text!r}")
    if base not in PRIMITIVES and base not in STRINGS:
        if base == "Header":
            base = "std_msgs/msg/Header"
        elif base.count("/") == 1:
            pkg, name = base.split("/")
            base = f"{pkg}/msg/{name}"
        elif base.count("/") == 0:
            base = f"{package}/msg/{base}"
    array, size = "", 0
    if m.group(0).endswith("]"):
        n = m.group("n")
        if m.group("le"):
            array, size = "bounded", int(n)
        elif n:
            array, size = "array", int(n)
        else:
            array = "sequence"
    return Type(base, sbound, array, size)


def parse_message(text: str, name: str) -> Message:
    """Parse the body of a .msg file (or one section of a .srv or .action file)."""
    package = name.split("/")[0]
    msg = Message(name)
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        type_text, _, rest = line.partition(" ")
        rest = rest.strip()
        if not rest:
            raise ValueError(f"{name}: missing field name in {raw!r}")
        ftype = parse_type(type_text, package)
        before, eq, value = rest.partition("=")
        if eq and _CONST_RE.match(before.strip()):
            msg.constants.append(Constant(before.strip(), ftype, value.strip()))
            continue
        fname, _, default = rest.partition(" ")
        if not _NAME_RE.match(fname):
            raise ValueError(f"{name}: bad field name {fname!r}")
        msg.fields.append(Field(fname, ftype, default.strip() or None))
    if not msg.fields:
        # rosidl gives every empty structure one placeholder member
        msg.fields.append(Field("structure_needs_at_least_one_member", Type("uint8")))
    return msg


def _split(text: str, parts: int, what: str) -> list[str]:
    sections, current = [], []
    for line in text.splitlines():
        if line.strip() == "---":
            sections.append("\n".join(current))
            current = []
        else:
            current.append(line)
    sections.append("\n".join(current))
    if len(sections) != parts:
        raise ValueError(f"{what}: expected {parts} sections separated by ---, found {len(sections)}")
    return sections


def _service(name: str, request: Message, response: Message) -> Service:
    """A service plus the ``_Event`` message rosidl adds for service introspection."""
    event = Message(
        f"{name}_Event",
        [
            Field("info", Type("service_msgs/msg/ServiceEventInfo")),
            Field("request", Type(request.name, array="bounded", size=1)),
            Field("response", Type(response.name, array="bounded", size=1)),
        ],
    )
    return Service(name, request, response, event)


def parse_service(text: str, name: str) -> Service:
    req, resp = _split(text, 2, name)
    return _service(name, parse_message(req, f"{name}_Request"), parse_message(resp, f"{name}_Response"))


def parse_action(text: str, name: str) -> Action:
    goal_t, result_t, feedback_t = _split(text, 3, name)
    goal = parse_message(goal_t, f"{name}_Goal")
    result = parse_message(result_t, f"{name}_Result")
    feedback = parse_message(feedback_t, f"{name}_Feedback")
    uuid = Type("unique_identifier_msgs/msg/UUID")
    send_goal = _service(
        f"{name}_SendGoal",
        Message(f"{name}_SendGoal_Request", [Field("goal_id", uuid), Field("goal", Type(goal.name))]),
        Message(f"{name}_SendGoal_Response", [Field("accepted", Type("bool")), Field("stamp", Type("builtin_interfaces/msg/Time"))]),
    )
    get_result = _service(
        f"{name}_GetResult",
        Message(f"{name}_GetResult_Request", [Field("goal_id", uuid)]),
        Message(f"{name}_GetResult_Response", [Field("status", Type("int8")), Field("result", Type(result.name))]),
    )
    feedback_message = Message(f"{name}_FeedbackMessage", [Field("goal_id", uuid), Field("feedback", Type(feedback.name))])
    return Action(name, goal, result, feedback, send_goal, get_result, feedback_message)


class Registry:
    """Finds and caches interface definitions under ``share/<pkg>/{msg,srv,action}/`` roots."""

    def __init__(self, roots: list[str | Path] | None = None) -> None:
        self.roots = [Path(r) for r in (roots if roots is not None else default_roots())]
        self.messages: dict[str, Message] = {}
        self.services: dict[str, Service] = {}
        self.actions: dict[str, Action] = {}

    def _find(self, pkg: str, kind: str, name: str) -> Path:
        for root in self.roots:
            path = root / pkg / kind / f"{name}.{kind}"
            if path.is_file():
                return path
        raise KeyError(f"{pkg}/{kind}/{name} not found under {[str(r) for r in self.roots]}")

    def add_text(self, full_name: str, text: str) -> None:
        """Register a definition from text, e.g. ``add_text("pkg/msg/Name", "int32 x")``."""
        pkg, kind, name = full_name.split("/")
        if kind == "msg":
            self.messages[full_name] = parse_message(text, full_name)
        elif kind == "srv":
            self._add_service(parse_service(text, full_name))
        elif kind == "action":
            self._add_action(parse_action(text, full_name))
        else:
            raise ValueError(f"unknown interface kind {kind!r}")

    def _add_service(self, srv: Service) -> None:
        self.services[srv.name] = srv
        for m in (srv.request, srv.response, srv.event):
            self.messages[m.name] = m

    def _add_action(self, act: Action) -> None:
        self.actions[act.name] = act
        for m in (act.goal, act.result, act.feedback, act.feedback_message):
            self.messages[m.name] = m
        self._add_service(act.send_goal)
        self._add_service(act.get_result)

    def message(self, name: str) -> Message:
        if name not in self.messages:
            self._ensure(name)
        return self.messages[name]

    def _ensure(self, name: str) -> None:
        """Load whichever file defines ``name``: a message, a service part, or an action part."""
        pkg, kind, short = name.split("/")
        if kind == "msg":
            self.add_text(name, self._find(pkg, "msg", short).read_text())
        elif kind == "srv":
            self.service(f"{pkg}/srv/{_SRV_PART.sub('', short)}")
        elif kind == "action":
            self.action(f"{pkg}/action/{_ACTION_PART.sub('', short)}")
        else:
            raise ValueError(f"unknown interface kind in {name!r}")

    def service(self, name: str) -> Service:
        if name not in self.services:
            pkg, _, short = name.split("/")
            if name.split("/")[1] == "action":
                self.action(f"{pkg}/action/{_ACTION_PART.sub('', short)}")
            else:
                self.add_text(name, self._find(pkg, "srv", short).read_text())
        return self.services[name]

    def action(self, name: str) -> Action:
        if name not in self.actions:
            pkg, _, short = name.split("/")
            self.add_text(name, self._find(pkg, "action", short).read_text())
        return self.actions[name]

    # ------------------------------------------------------- type hashes --

    def _individual(self, name: str, kind: str = "msg") -> dict:
        if kind == "srv":
            srv = self.service(name)
            fields = [
                ("request_message", Type(srv.request.name)),
                ("response_message", Type(srv.response.name)),
                ("event_message", Type(srv.event.name)),
            ]
        elif kind == "action":
            act = self.action(name)
            fields = [
                ("goal", Type(act.goal.name)),
                ("result", Type(act.result.name)),
                ("feedback", Type(act.feedback.name)),
                ("send_goal_service", Type(act.send_goal.name)),
                ("get_result_service", Type(act.get_result.name)),
                ("feedback_message", Type(act.feedback_message.name)),
            ]
        else:
            fields = [(f.name, f.type) for f in self.message(name).fields]
        return {
            "type_name": name,
            "fields": [
                {
                    "name": fname,
                    "type": {
                        "type_id": ftype.type_id(),
                        "capacity": ftype.size if ftype.array else 0,
                        "string_capacity": ftype.string_bound,
                        "nested_type_name": ftype.base if ftype.is_nested else "",
                    },
                }
                for fname, ftype in fields
            ],
        }

    def _kind_of(self, name: str) -> str:
        """Services and actions appear as nested types inside the service and action descriptions."""
        if name in self.services:
            return "srv"
        if name in self.actions:
            return "action"
        return "msg"

    def type_description(self, name: str) -> dict:
        """The hashable REP 2011 description: the type plus every type it references, sorted."""
        if name not in self.services and name not in self.actions and name not in self.messages:
            self._ensure(name)
        top = self._individual(name, self._kind_of(name))
        seen: dict[str, dict] = {}
        queue = [f["type"]["nested_type_name"] for f in top["fields"] if f["type"]["nested_type_name"]]
        while queue:
            ref = queue.pop()
            if ref in seen:
                continue
            desc = self._individual(ref, self._kind_of(ref))
            seen[ref] = desc
            queue.extend(f["type"]["nested_type_name"] for f in desc["fields"] if f["type"]["nested_type_name"])
        return {"type_description": top, "referenced_type_descriptions": [seen[k] for k in sorted(seen)]}

    def type_hash(self, name: str) -> str:
        """``RIHS01_`` plus the SHA-256 of the description serialized the way REP 2011 says."""
        text = json.dumps(
            self.type_description(name), ensure_ascii=True, indent=None, separators=(", ", ": "), sort_keys=False, allow_nan=False
        )
        return "RIHS01_" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def show(self, name: str, indent: str = "") -> str:
        """Render a message like ``ros2 interface show --no-comments`` does, nested types expanded."""
        lines = []
        msg = self.message(name)
        for c in msg.constants:
            lines.append(f"{indent}{c.type} {c.name}={c.value}")
        for f in msg.fields:
            if f.name == "structure_needs_at_least_one_member":
                continue
            default = f" {f.default}" if f.default is not None else ""
            lines.append(f"{indent}{f.type} {f.name}{default}")
            if f.type.is_nested:
                lines.append(self.show(f.type.base, indent + "\t"))
        return "\n".join(line for line in lines if line)


def default_roots() -> list[Path]:
    """The interfaces bundled with this package, the repository workspace, then AMENT_PREFIX_PATH."""
    import os

    roots = [Path(__file__).parent / "interfaces"]
    workspace = Path(__file__).resolve().parents[1] / "ros2_ws" / "src"
    if workspace.is_dir():
        roots.append(workspace)  # this repository's own packages (ugs_interfaces, tutorial_interfaces)
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix:
            roots.append(Path(prefix) / "share")
    return roots
