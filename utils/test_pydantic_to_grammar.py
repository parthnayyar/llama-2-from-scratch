from __future__ import annotations
import unittest
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pydantic_to_grammar import PydanticToLark, _PRIMITIVE_MAP, _LARK_HELPERS, _LARK_IMPORTS

class TestPydanticToLark(unittest.TestCase):
    
    def test_primitive_types(self):
        """Test basic primitive type conversion"""
        
        class SimpleModel(BaseModel):
            name: str
            age: int
            score: float
            active: bool
        
        converter = PydanticToLark(SimpleModel)
        grammar = converter.to_lark()
        
        # Check that primitive mappings are used
        self.assertIn("ESCAPED_STRING", grammar)
        self.assertIn("SIGNED_NUMBER", grammar)
        self.assertIn("\"true\"|\"false\"", grammar)
        
        # Check structure
        self.assertIn("start:", grammar)
        self.assertIn("%import common.ESCAPED_STRING", grammar)
        self.assertIn("%ignore WS", grammar)
    
    def test_nested_objects(self):
        """Test nested object conversion"""
        
        class Address(BaseModel):
            street: str
            city: str
            zip_code: str
        
        class Person(BaseModel):
            name: str
            address: Address
        
        converter = PydanticToLark(Person)
        grammar = converter.to_lark()
        
        # Should contain rules for both Person and Address
        self.assertIn("person:", grammar)
        self.assertIn("address:", grammar)
        
        # Should have proper object structure with braces
        self.assertIn("\"{", grammar)
        self.assertIn("}\"", grammar)
    
    def test_arrays_and_lists(self):
        """Test array/list conversion"""
        
        class ListModel(BaseModel):
            tags: list[str]
            scores: list[float]
        
        converter = PydanticToLark(ListModel)
        grammar = converter.to_lark()
        
        # Should contain array syntax
        self.assertIn("\"[\"", grammar)
        self.assertIn("\"]\"", grammar)
        
        # Should have list item rules
        self.assertTrue(any("_item" in rule for rule in converter.rules.keys()))
        self.assertTrue(any("pair_list_" in rule for rule in converter.rules.keys()))
    
    def test_tuples(self):
        """Test fixed-length tuple conversion"""
        
        class TupleModel(BaseModel):
            coordinates: tuple[float, float]
            rgb: tuple[int, int, int]
        
        converter = PydanticToLark(TupleModel)
        grammar = converter.to_lark()
        
        # Should contain tuple rules with specific element counts
        self.assertTrue(any("_t0" in rule for rule in converter.rules.keys()))
        self.assertTrue(any("_t1" in rule for rule in converter.rules.keys()))
        self.assertTrue(any("_t2" in rule for rule in converter.rules.keys()))
    
    def test_optional_fields(self):
        """Test optional field handling"""
        
        class OptionalModel(BaseModel):
            required_field: str
            optional_field: Optional[str]  # Optional but no default value
        
        converter = PydanticToLark(OptionalModel)
        grammar = converter.to_lark()
        
        # Should have optional field rules
        self.assertTrue(any("opt_" in rule for rule in converter.rules.keys()))
        self.assertIn("optional_field", grammar)
    
    def test_unions(self):
        """Test union type conversion"""
        
        class UnionModel(BaseModel):
            value: str | int
            data: dict | list
        
        converter = PydanticToLark(UnionModel)
        grammar = converter.to_lark()
        
        # Should contain union rules
        self.assertTrue(any("_u0" in rule for rule in converter.rules.keys()))
        self.assertTrue(any("_u1" in rule for rule in converter.rules.keys()))
        self.assertIn("|", grammar)
    
    def test_name_sanitization(self):
        """Test rule name sanitization"""
        # Create a minimal model to initialize the converter
        class TestModel(BaseModel):
            field: str
        
        converter = PydanticToLark(TestModel)
        
        # Test various invalid characters
        test_cases = [
            ("valid_name", "valid_name"),
            ("Invalid-Name", "invalid_name"),
            ("123invalid", "_123invalid"),
            ("with spaces", "with_spaces"),
            ("with.dots", "with_dots"),
            ("", "unnamed"),
            ("special!@#chars", "special___chars")
        ]
        
        for input_name, expected in test_cases:
            result = converter._sanitize_name(input_name)
            self.assertEqual(result, expected)
    
    def test_complex_nested_structure(self):
        """Test complex nested data structures"""
        
        class Config(BaseModel):
            debug: bool
            max_retries: int
        
        class Database(BaseModel):
            host: str
            port: int
            config: Config
        
        class Application(BaseModel):
            name: str
            version: str
            databases: list[Database]
            metadata: dict[str, str | int]
        
        converter = PydanticToLark(Application)
        grammar = converter.to_lark()
        
        # Should handle deep nesting
        self.assertIn("application:", grammar)
        self.assertIn("database:", grammar)
        self.assertIn("config:", grammar)
        
        # Should be a valid grammar string
        self.assertIsInstance(grammar, str)
        self.assertTrue(len(grammar) > 100)  # Should be substantial
    
    def test_schema_with_refs(self):
        """Test handling of schema references ($ref)"""
        
        class Node(BaseModel):
            value: str
            # Use a forward reference but no default value to avoid the assertion
            children: Optional[list[Node]]
        
        # Update the Node model to resolve the forward reference
        Node.model_rebuild()
        
        converter = PydanticToLark(Node)
        grammar = converter.to_lark()
        
        # Should handle recursive references without infinite loops
        self.assertIn("node:", grammar)
        self.assertIsInstance(grammar, str)
    
    def test_empty_model(self):
        """Test empty model conversion"""
        
        class EmptyModel(BaseModel):
            pass
        
        converter = PydanticToLark(EmptyModel)
        grammar = converter.to_lark()
        
        # Should create valid grammar even for empty models
        self.assertIn("start:", grammar)
        self.assertIn("emptymodel:", grammar)
        self.assertIn("\"{\" \"}\"", grammar)  # Empty object
    
    def test_model_with_field_names_requiring_sanitization(self):
        """Test models with field names that need sanitization"""
        
        class WeirdFieldNames(BaseModel):
            normal_field: str
            field_with_dashes: str = Field(alias="field-with-dashes")
            field_with_spaces: str = Field(alias="field with spaces")
            field_123: str = Field(alias="123field")
        
        converter = PydanticToLark(WeirdFieldNames)
        grammar = converter.to_lark()
        
        # Should handle field name sanitization
        self.assertIsInstance(grammar, str)
        self.assertIn("start:", grammar)
    
    def test_constants_and_helpers(self):
        """Test that constants and helpers are properly defined"""
        
        # Test primitive map
        self.assertIsInstance(_PRIMITIVE_MAP, dict)
        self.assertIn("string", _PRIMITIVE_MAP)
        self.assertIn("integer", _PRIMITIVE_MAP)
        self.assertIn("boolean", _PRIMITIVE_MAP)
        
        # Test helper strings
        self.assertIsInstance(_LARK_HELPERS, str)
        self.assertIn("json_value", _LARK_HELPERS)
        
        self.assertIsInstance(_LARK_IMPORTS, str)
        self.assertIn("%import", _LARK_IMPORTS)
        self.assertIn("%ignore", _LARK_IMPORTS)
    
    def test_to_lark_with_custom_root_rule(self):
        """Test custom root rule naming"""
        
        class TestModel(BaseModel):
            name: str
        
        converter = PydanticToLark(TestModel)
        grammar = converter.to_lark(root_rule="custom_root")
        
        self.assertIn("start: custom_root", grammar)
        self.assertIn("custom_root:", grammar)
    
    def test_model_json_schema_method(self):
        """Test the model_to_json_schema method"""
        
        class TestModel(BaseModel):
            name: str
            age: int
        
        converter = PydanticToLark(TestModel)
        
        # Test with class
        schema1 = converter.model_to_json_schema(TestModel)
        self.assertIsInstance(schema1, dict)
        self.assertIn("properties", schema1)
        
        # Test with instance
        instance = TestModel(name="test", age=25)
        schema2 = converter.model_to_json_schema(instance)
        self.assertIsInstance(schema2, dict)
        self.assertIn("properties", schema2)
        
        # Schemas should be the same
        self.assertEqual(schema1, schema2)
    
    def test_literal_types(self):
        """Test Literal type handling"""
        
        class LiteralModel(BaseModel):
            status: Literal["active", "inactive", "pending"]
            level: Literal[1, 2, 3]
            flag: Literal[True, False]
        
        converter = PydanticToLark(LiteralModel)
        grammar = converter.to_lark()
        
        # Should contain literal values
        self.assertIn("\"active\"", grammar)
        self.assertIn("1", grammar)
        self.assertIn("true", grammar)
        self.assertIn("|", grammar)  # Union operator
    
    def test_unsupported_default_values(self):
        """Test assertion for unsupported default values"""
        
        # Define a model with default values
        try:
            class ModelWithDefaults(BaseModel):
                name: str = "default"
            
            PydanticToLark(ModelWithDefaults)
            self.fail("Should have raised an assertion error for default values")
        except AssertionError as e:
            self.assertIn("Default values not supported", str(e))
            self.assertIn("name", str(e))

if __name__ == "__main__":
    unittest.main()