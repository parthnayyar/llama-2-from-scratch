from __future__ import annotations
from pydantic import BaseModel
from typing import Type, Any, Literal
import json
import re

# Type alias for JSON-like structure
_JSON = dict[str, Any|dict]

class PydanticToLark:
    def __init__(self, model: BaseModel|Type[BaseModel]) -> None:
        self.schema: _JSON = self.model_to_json_schema(model)
        # Check for unsupported defaults
        for field_name, field_schema in self.schema.get("properties", {}).items():
            assert "default" not in field_schema, f"Default values not supported for field '{field_name}'"
        
        self.defs: _JSON = self.schema.get("$defs", {})
        self.rules: dict[str, str] = {}
        self.seen: set[str] = set()

    def model_to_json_schema(self, model: BaseModel|Type[BaseModel]) -> _JSON:
        """
        Returns a Draft-2020-12 compatible JSON schema for a Pydantic model.
        Works for both classes and instances.
        """
        m_cls: Type[BaseModel] = model if isinstance(model, type) else model.__class__
        
        # Check that model is a Pydantic model
        assert issubclass(m_cls, BaseModel), f"Expected Pydantic model, got {m_cls.__name__}"
        
        return m_cls.model_json_schema()
        
    def to_lark(self, root_rule: str|None = None) -> str:
        """
        Convert the schema to a Lark grammar.
        
        Args:
            root_rule: Optional name of the root rule. If not provided, the schema title or "root" is used.
        
        Returns:
            A string representing a Lark grammar
        """
        # Generate a valid identifier for the root rule
        root_rule: str = self._sanitize_name(root_rule or self.schema.get("title") or "root")
        
        # Process the schema to generate rules
        self._rule_for_schema(root_rule, self.schema)
        
        # Combine all rules
        model_rules: str = "\n".join(f"{k}: {v}" for k, v in self.rules.items())
        
        # Define the start rule
        start_rule: str = f"start: {root_rule}"
        
        # Add imports and helpers
        return f"{_LARK_IMPORTS}\n\n{start_rule}\n\n{model_rules}\n\n{_LARK_HELPERS}"
    
    def _sanitize_name(self, name: str) -> str:
        """
        Convert a string to a valid Lark rule name by replacing invalid characters.
        """
        # Replace non-alphanumeric characters with underscores
        sanitized: str = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Ensure the name starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed"
        return sanitized.lower()
    
    def _rule_for_schema(self, name: str, node: _JSON) -> str:
        """
        Recursively generate rules for a schema node.
        
        Args:
            name: Name for the rule being generated
            node: Schema node to process
            
        Returns:
            Name of the generated rule
        """
        name: str = self._sanitize_name(name)
        
        # Avoid infinite recursion
        if name in self.seen:
            return name
        self.seen.add(name)
        
        # Handle references (needed for recursive types)
        if "$ref" in node:
            ref_name: str = node["$ref"].split("/")[-1]
            ref_schema: _JSON = self.defs[ref_name]
            return self._rule_for_schema(ref_name, ref_schema)
        
        node_type: str|list[str] = node.get("type")
        rule_body: str
        
        # Handle different types
        if node_type == "object":
            rule_body = self._object_rule(node, name)
        
        elif node_type == "array":
            # Check if it's a tuple (fixed-length array)
            if (
                "prefixItems" in node
                and isinstance(node["prefixItems"], list)
                and node.get("minItems") == node.get("maxItems") == len(node["prefixItems"])
                and (node.get("items") in (False, {}, None))  # no extras allowed
            ):
                rule_body = self._tuple_rule(node, name)
            else:
                # Regular array/list
                items = node.get("items", {})
                item_rule_name: str = self._rule_for_schema(f"{name}_item", items)
                rule_body = f'"[" [pair_list_{name}] "]"'
                
                # Add array item list rule
                pair_list_name: str = f"pair_list_{name}"
                pair_list_rule: str = f'{item_rule_name} ("," {item_rule_name})*'
                self.rules[pair_list_name] = pair_list_rule
        
        # Handle primitive types
        elif node_type in ("string", "integer", "number", "boolean", "null"):
            assert node_type in _PRIMITIVE_MAP, f"Unsupported primitive type: {node_type}"
            rule_body = _PRIMITIVE_MAP[node_type]
        
        # Handle enums (for Literal types)
        elif "enum" in node:
            enum_values: list[str] = []
            for v in node["enum"]:
                if isinstance(v, str):
                    # String values need quotes
                    enum_values.append(f'"{v}"')
                elif isinstance(v, (int, float, bool, type(None))):
                    # Supported literal types
                    enum_values.append(json.dumps(v))
                else:
                    # Unsupported literal types
                    assert False, f"Unsupported literal value type: {type(v).__name__}"
            
            rule_body = " | ".join(enum_values)
        
        # Handle unions
        elif "anyOf" in node or isinstance(node_type, list):
            sub_schemas: list[dict] = node.get("anyOf") or [{"type": t} for t in node_type]
            alt_rules: list[str] = [
                self._rule_for_schema(f"{name}_u{i}", s) for i, s in enumerate(sub_schemas)
            ]
            rule_body = " | ".join(alt_rules)
        
        # Unsupported type
        else:
            assert False, f"Unsupported schema type for '{name}': {node}"
        
        self.rules[name] = rule_body
        return name
    
    def _tuple_rule(self, node: _JSON, name: str) -> str:
        """
        Generate a rule for a fixed-length tuple.
        """
        elem_rules: list[str] = []
        for i, elem_schema in enumerate(node["prefixItems"]):
            elem_rule_name: str = self._rule_for_schema(f"{name}_t{i}", elem_schema)
            elem_rules.append(elem_rule_name)
        
        joined_elems: str = " \",\" ".join(elem_rules)
        return f'"[" {joined_elems} "]"'
    
    def _object_rule(self, node: _JSON, name: str) -> str:
        """
        Generate a rule for an object with properties.
        """
        required: set[str] = set(node.get("required", []))
        pairs: list[str] = []
        pair_rules: dict[str, str] = {}
        
        for field, spec in node.get("properties", {}).items():
            field_name: str = self._sanitize_name(f"{name}_{field}")
            field_rule: str = self._rule_for_schema(field_name, spec)
            
            # Create a rule for this specific field pair
            pair_name: str = f"pair_{name}_{field}"
            pair_rule: str = f'"{field}" ":" {field_rule}'
            pair_rules[pair_name] = pair_rule
            
            if field in required:
                pairs.append(pair_name)
            else:
                # Optional fields
                opt_name: str = f"opt_{name}_{field}"
                self.rules[opt_name] = f"{pair_name}?"
                pairs.append(opt_name)
        
        # Add all the pair rules
        self.rules.update(pair_rules)
        
        # If no pairs, just an empty object
        if not pairs:
            return '"{" "}"'
        
        # Create a rule for the object fields
        obj_content_name: str = f"obj_content_{name}"
        pairs_joined: str = " \",\" ".join(pairs)
        self.rules[obj_content_name] = pairs_joined
        
        return f'"{{"  {obj_content_name} "}}"'

# Mapping of JSON Schema primitive types to Lark rules
_PRIMITIVE_MAP: dict[str, str] = {
    "string": "ESCAPED_STRING",
    "integer": "SIGNED_NUMBER",
    "number": "SIGNED_NUMBER",
    "boolean": '"true"|"false"',
    "null": '"null"',
}

# Base helper rules for Lark grammar
_LARK_HELPERS: str = """
// Helper rules for JSON elements
json_value: ESCAPED_STRING
          | SIGNED_NUMBER
          | object
          | array
          | "true" 
          | "false"
          | "null"

object: "{" [pair ("," pair)*] "}"
pair: ESCAPED_STRING ":" json_value
array: "[" [json_value ("," json_value)*] "]"
"""

_LARK_IMPORTS: str = """
// Import common terminal definitions
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

// Ignore whitespace
%ignore WS
"""

def pydantic_to_lark_grammar(model: Type[BaseModel]) -> str:
    """
    Convert a Pydantic model to a Lark grammar string.
    
    Args:
        model: A Pydantic model class
        
    Returns:
        A string containing the Lark grammar
    """
    converter = PydanticToLark(model)
    return converter.to_lark()