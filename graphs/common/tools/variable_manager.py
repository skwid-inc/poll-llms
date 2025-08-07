"""
Variable Manager Tool - Manages dynamic variables in call state
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, Field, create_model
from langchain.tools import tool
from langchain_core.tools import StructuredTool

from app_config import AppConfig
from graphs.common.agent_state import StatefulBaseModel

logger = logging.getLogger(__name__)


# Default schema for backwards compatibility
class VariableManagerRequest(StatefulBaseModel):
    """Default schema for variable manager tool"""

    action: str = Field(
        description='Action to perform. Must be EXACTLY one of: "get_status", "update_variable"'
    )
    variable_name: Optional[str] = Field(
        None,
        description="Name of the variable to update (required for update_variable action)",
    )
    variable_value: Optional[Union[str, int, float, bool]] = Field(
        None,
        description="Value to set for the variable (required for update_variable action)",
    )
    notes: Optional[str] = Field(
        None, description="Optional notes about the update or context"
    )


def create_variable_manager_request_schema(
    config: Dict[str, Any],
) -> Type[BaseModel]:
    """Create a dynamic schema for variable manager requests based on available variables."""

    # Get all available variables from config
    required_fields = config.get("required_fields", [])
    optional_fields = config.get("optional_fields", [])
    all_fields = required_fields + optional_fields

    # Create field descriptions
    field_descriptions = config.get("field_descriptions", {})

    # Create the base fields that are always present
    fields = {
        "action": (
            str,
            Field(
                description='Action to perform. Must be EXACTLY one of: "get_status", "update_variable"'
            ),
        ),
        "variable_name": (
            Optional[str],
            Field(
                None,
                description=f"Name of the variable to update (required for update_variable action). Must be EXACTLY one of: {', '.join(f'"{field}"' for field in all_fields)}",
            ),
        ),
        "variable_value": (
            Optional[Union[str, int, float, bool]],
            Field(
                None,
                description="Value to set for the variable (required for update_variable action)",
            ),
        ),
        "notes": (
            Optional[str],
            Field(
                None, description="Optional notes about the update or context"
            ),
        ),
    }

    # Create the dynamic model
    return create_model("VariableManagerRequest", **fields)


def get_base_variable_config() -> Dict[str, Any]:
    """Get the base variable configuration from call_config or defaults."""
    # Try to get base config from call_config first
    if AppConfig().call_config and hasattr(
        AppConfig().call_config, "variables"
    ):
        return AppConfig().call_config.variables

    logger.info(
        f"FALLING BACK TO DEFAULT VARIABLE CONFIG: {AppConfig().call_config}"
    )
    # Default fallback configuration
    return {
        "required_fields": [],
        "optional_fields": [],
        "field_descriptions": {},
        "field_validation": {},
        "completion_threshold": 1.0,
    }


def transform_validation_rules(
    validation_config: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Transform validation_config list into field_validation dict format."""
    field_validation = {}

    for rule in validation_config:
        var_name = rule["variable_name"]
        data_type = rule.get(
            "data_type", "string"
        )  # Default to string if not specified
        validation_type = rule["validation_type"]
        comparison_type = rule["comparison_type"]
        comparison_value = rule["comparison_value"]
        error_message = rule.get("error_message", "")
        format_pattern = rule.get("format", "")

        # Initialize field validation if not exists
        if var_name not in field_validation:
            field_validation[var_name] = {}

        # Set the data type based on the rule
        field_validation[var_name]["type"] = data_type

        # Transform rule based on validation_type and data_type
        if validation_type == "required":
            field_validation[var_name]["required"] = True

        elif validation_type == "greater_than":
            if comparison_type == "hardcoded":
                if data_type == "number":
                    field_validation[var_name]["min"] = float(comparison_value)
                elif data_type == "date":
                    field_validation[var_name]["after_date"] = comparison_value
            elif comparison_type == "variable":
                field_validation[var_name]["compare_variable"] = (
                    comparison_value
                )
                field_validation[var_name]["compare_operator"] = "greater_than"

        elif validation_type == "less_than":
            if comparison_type == "hardcoded":
                if data_type == "number":
                    field_validation[var_name]["max"] = float(comparison_value)
                elif data_type == "date":
                    field_validation[var_name]["before_date"] = comparison_value
            elif comparison_type == "variable":
                field_validation[var_name]["compare_variable"] = (
                    comparison_value
                )
                field_validation[var_name]["compare_operator"] = "less_than"

        elif validation_type == "equals":
            if comparison_type == "hardcoded":
                if data_type == "boolean":
                    field_validation[var_name]["exact_value"] = (
                        comparison_value.lower() == "true"
                    )
                else:
                    field_validation[var_name]["exact_value"] = comparison_value
            elif comparison_type == "variable":
                field_validation[var_name]["compare_variable"] = (
                    comparison_value
                )
                field_validation[var_name]["compare_operator"] = "equals"

        elif validation_type == "not_equals":
            if comparison_type == "hardcoded":
                if data_type == "boolean":
                    field_validation[var_name]["not_equal_value"] = (
                        comparison_value.lower() == "true"
                    )
                else:
                    field_validation[var_name]["not_equal_value"] = (
                        comparison_value
                    )
            elif comparison_type == "variable":
                field_validation[var_name]["compare_variable"] = (
                    comparison_value
                )
                field_validation[var_name]["compare_operator"] = "not_equals"

        elif validation_type == "min_length":
            field_validation[var_name]["min_length"] = int(comparison_value)

        elif validation_type == "max_length":
            field_validation[var_name]["max_length"] = int(comparison_value)

        elif validation_type == "format":
            if data_type == "date":
                field_validation[var_name]["date_format"] = (
                    format_pattern or comparison_value
                )
            else:  # string regex
                field_validation[var_name]["regex_pattern"] = (
                    format_pattern or comparison_value
                )

        # Set error message (last one wins if multiple rules for same field)
        if error_message:
            field_validation[var_name]["error_message"] = error_message

    return field_validation


def create_manage_variables_tool(
    validation_config: Optional[List[Dict[str, Any]]] = None,
) -> StructuredTool:
    """
    Factory function to create a manage_variables tool with specific validation config.

    Args:
        validation_config: Optional validation configuration. If None, uses default config.

    Returns:
        A StructuredTool configured with the provided validation config.
    """

    # Get base config from call_config (required_fields, optional_fields, field_descriptions, etc.)
    base_config = get_base_variable_config()

    # If no validation_config provided, try to get it from call_config
    if (
        validation_config is None
        and AppConfig().call_config
        and hasattr(AppConfig().call_config, "validation_config")
    ):
        validation_config = AppConfig().call_config.validation_config
        logger.info(
            f"Using validation_config from call_config: {validation_config}"
        )

    # Transform validation rules and merge with base config
    if validation_config:
        field_validation = transform_validation_rules(validation_config)
        # Merge with existing field_validation from base config
        existing_validation = base_config.get("field_validation", {})
        merged_validation = {**existing_validation, **field_validation}
        config = {**base_config, "field_validation": merged_validation}
    else:
        config = base_config

    logger.info(f"Creating manage_variables tool with config: {config}")

    # Create dynamic schema based on config
    VariableManagerRequest = create_variable_manager_request_schema(config)

    def calculate_completion_status(
        variables: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate completion status based on current variables and configuration"""

        required_fields = config.get("required_fields", [])
        optional_fields = config.get("optional_fields", [])

        # Check required fields
        completed_required = []
        missing_required = []

        for field in required_fields:
            value = variables.get(field)
            if value is not None and value != "" and value != "null":
                completed_required.append(field)
            else:
                missing_required.append(field)

        # Check optional fields
        completed_optional = []
        missing_optional = []

        for field in optional_fields:
            value = variables.get(field)
            if value is not None and value != "" and value != "null":
                completed_optional.append(field)
            else:
                missing_optional.append(field)

        # Calculate percentages
        total_required = len(required_fields)
        completed_required_count = len(completed_required)

        required_completion_rate = (
            completed_required_count / total_required
            if total_required > 0
            else 1.0
        )

        total_optional = len(optional_fields)
        completed_optional_count = len(completed_optional)

        optional_completion_rate = (
            completed_optional_count / total_optional
            if total_optional > 0
            else 1.0
        )

        # Overall completion
        overall_completion_rate = required_completion_rate

        # Determine if call can be completed
        completion_threshold = config.get("completion_threshold", 1.0)
        can_complete = required_completion_rate >= completion_threshold

        return {
            "required_fields": {
                "total": total_required,
                "completed": completed_required_count,
                "completion_rate": required_completion_rate,
                "completed_fields": completed_required,
                "missing_fields": missing_required,
            },
            "optional_fields": {
                "total": total_optional,
                "completed": completed_optional_count,
                "completion_rate": optional_completion_rate,
                "completed_fields": completed_optional,
                "missing_fields": missing_optional,
            },
            "overall": {
                "completion_rate": overall_completion_rate,
                "can_complete": can_complete,
                "next_required": missing_required[0]
                if missing_required
                else None,
            },
        }

    def manage_variables_impl(
        action: str,
        variable_name: Optional[str] = None,
        variable_value: Optional[Union[str, int, float, bool]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Implementation of manage_variables with baked-in validation config.
        """

        try:
            # Get current call metadata and variables
            call_metadata = AppConfig().get_call_metadata()
            if not call_metadata:
                return "❌ Error: No call metadata found"

            # Initialize variables if they don't exist
            if "variables" not in call_metadata:
                call_metadata["variables"] = {}
                AppConfig().set_call_metadata(call_metadata)

            current_variables = call_metadata.get("variables", {})

            def validate_variable_value(
                field_name: str, value: Any
            ) -> tuple[bool, str, Any]:
                """Validate a variable value against its validation rules and return normalized value"""
                logger.info(
                    f"Starting validation for field '{field_name}' with value: {value}"
                )

                # Check if the variable name is valid
                all_fields = config.get("required_fields", []) + config.get(
                    "optional_fields", []
                )
                if field_name not in all_fields:
                    return (
                        False,
                        f"'{field_name}' is not a valid variable name. Available variables are: {', '.join(all_fields)}",
                        value,
                    )

                validation_rules = config.get("field_validation", {}).get(
                    field_name
                )
                if not validation_rules:
                    logger.info(
                        f"No validation rules found for field '{field_name}', skipping validation"
                    )
                    return True, "", value

                data_type = validation_rules.get("type", "string")
                logger.info(f"Field '{field_name}' has data type: {data_type}")

                # Check required field
                if validation_rules.get("required") and (
                    value is None or value == "" or value == "null"
                ):
                    error_msg = validation_rules.get(
                        "error_message", f"{field_name} is required"
                    )
                    logger.warning(
                        f"Required field '{field_name}' is empty or null"
                    )
                    return False, error_msg, value

                # Skip other validations if value is empty (optional field)
                if value is None or value == "" or value == "null":
                    logger.info(
                        f"Optional field '{field_name}' is empty, skipping validation"
                    )
                    return True, "", value

                if data_type == "date":
                    logger.info(
                        f"Validating date field '{field_name}' with value: '{value}'"
                    )

                    def convert_format_to_strftime(format_str: str) -> str:
                        """Convert common format strings to strftime format"""
                        conversions = {
                            "MM/DD/YYYY": "%m/%d/%Y",
                            "DD/MM/YYYY": "%d/%m/%Y",
                            "YYYY-MM-DD": "%Y-%m-%d",
                            "DD-MM-YYYY": "%d-%m-%Y",
                            "YYYY/MM/DD": "%Y/%m/%d",
                            "MM-DD-YYYY": "%m-%d-%Y",
                        }
                        # If it's already a strftime format (contains %), return as-is
                        if "%" in format_str:
                            return format_str
                        # Otherwise try to convert it
                        return conversions.get(format_str, format_str)

                    def parse_flexible_date(
                        date_str: str, target_format: str
                    ) -> tuple[bool, str, str]:
                        """
                        Try to parse date in multiple common formats and convert to target format.
                        Returns (success, error_message, formatted_date)
                        """
                        from datetime import datetime

                        # Convert the target format to strftime format if needed
                        strftime_format = convert_format_to_strftime(
                            target_format
                        )

                        # Common date formats to try (in order of preference)
                        common_formats = [
                            strftime_format,  # Try target format first
                            "%Y-%m-%d %H:%M:%S",  # 2024-03-15 14:30:00
                            "%Y-%m-%d %H:%M",  # 2024-03-15 14:30
                            "%Y-%m-%d",  # 2024-03-15
                            "%m/%d/%Y %H:%M",  # 03/15/2024 14:30
                            "%m/%d/%Y",  # 03/15/2024
                            "%m-%d-%Y %H:%M",  # 03-15-2024 14:30
                            "%m-%d-%Y",  # 03-15-2024
                            "%d/%m/%Y %H:%M",  # 15/03/2024 14:30
                            "%d/%m/%Y",  # 15/03/2024
                            "%B %d, %Y %H:%M",  # March 15, 2024 14:30
                            "%B %d, %Y",  # March 15, 2024
                            "%b %d, %Y %H:%M",  # Mar 15, 2024 14:30
                            "%b %d, %Y",  # Mar 15, 2024
                            "%d %B %Y %H:%M",  # 15 March 2024 14:30
                            "%d %B %Y",  # 15 March 2024
                            "%d %b %Y %H:%M",  # 15 Mar 2024 14:30
                            "%d %b %Y",  # 15 Mar 2024
                            "%Y/%m/%d %H:%M",  # 2024/03/15 14:30
                            "%Y/%m/%d",  # 2024/03/15
                        ]

                        parsed_date = None
                        for fmt in common_formats:
                            try:
                                parsed_date = datetime.strptime(
                                    date_str.strip(), fmt
                                )
                                logger.info(
                                    f"Successfully parsed '{date_str}' using format '{fmt}'"
                                )
                                break
                            except ValueError:
                                continue

                        if parsed_date is None:
                            # If we can't parse it, check if it's missing critical components
                            date_str_lower = date_str.lower().strip()

                            # Check for incomplete dates (missing year is most critical)
                            if any(
                                word in date_str_lower
                                for word in ["yesterday", "today", "tomorrow"]
                            ):
                                return (
                                    False,
                                    "Please provide the specific date instead of relative terms like 'yesterday', 'today', or 'tomorrow'",
                                    date_str,
                                )

                            if re.match(
                                r"^\d{1,2}[/-]\d{1,2}$", date_str.strip()
                            ):  # MM/DD or MM-DD without year
                                return (
                                    False,
                                    "Please include the year in the date (e.g., '03/15/2024')",
                                    date_str,
                                )

                            if re.match(
                                r"^[a-zA-Z]+ \d{1,2}$", date_str.strip()
                            ):  # "March 15" without year
                                return (
                                    False,
                                    "Please include the year in the date (e.g., 'March 15, 2024')",
                                    date_str,
                                )

                            return (
                                False,
                                f"Could not parse date '{date_str}'. Please provide a clear date format like '2024-03-15' or 'March 15, 2024'",
                                date_str,
                            )

                        # Convert to target format
                        try:
                            formatted_date = parsed_date.strftime(
                                strftime_format
                            )
                            logger.info(
                                f"Converted date to target format '{target_format}' (strftime: '{strftime_format}'): {formatted_date}"
                            )
                            return True, "", formatted_date
                        except Exception as e:
                            logger.error(
                                f"Error formatting date to target format: {e}"
                            )
                            return (
                                False,
                                f"Error formatting date: {str(e)}",
                                date_str,
                            )

                    # Get target format (default to a reasonable format if not specified)
                    target_format = validation_rules.get(
                        "date_format", "%Y-%m-%d %H:%M"
                    )

                    # Try to parse and normalize the date
                    success, error_msg, normalized_date = parse_flexible_date(
                        str(value), target_format
                    )
                    if not success:
                        logger.warning(
                            f"Date parsing failed for '{field_name}': {error_msg}"
                        )
                        return (
                            False,
                            validation_rules.get("error_message", error_msg),
                            value,
                        )

                    # Use the normalized date for further validation and storage
                    parsed_value = normalized_date
                    logger.info(f"Using normalized date value: {parsed_value}")

                    # Date comparison validation (using the parsed datetime object for comparisons)
                    if (
                        "after_date" in validation_rules
                        or "before_date" in validation_rules
                    ):
                        try:
                            from datetime import datetime

                            # Convert target format to strftime format for parsing
                            strftime_format = convert_format_to_strftime(
                                target_format
                            )

                            value_date = datetime.strptime(
                                parsed_value, strftime_format
                            )

                            if "after_date" in validation_rules:
                                logger.info(
                                    f"Checking if date is after: {validation_rules['after_date']}"
                                )
                                after_date = datetime.strptime(
                                    validation_rules["after_date"],
                                    strftime_format,
                                )
                                if value_date <= after_date:
                                    logger.warning(
                                        f"Date '{parsed_value}' is not after required date '{validation_rules['after_date']}'"
                                    )
                                    return (
                                        False,
                                        validation_rules.get(
                                            "error_message",
                                            f"Date must be after {validation_rules['after_date']}",
                                        ),
                                        value,
                                    )

                            if "before_date" in validation_rules:
                                logger.info(
                                    f"Checking if date is before: {validation_rules['before_date']}"
                                )
                                before_date = datetime.strptime(
                                    validation_rules["before_date"],
                                    strftime_format,
                                )
                                if value_date >= before_date:
                                    logger.warning(
                                        f"Date '{parsed_value}' is not before required date '{validation_rules['before_date']}'"
                                    )
                                    return (
                                        False,
                                        validation_rules.get(
                                            "error_message",
                                            f"Date must be before {validation_rules['before_date']}",
                                        ),
                                        value,
                                    )

                        except ValueError as e:
                            logger.warning(
                                f"Error during date comparison validation for '{field_name}': {e}"
                            )
                            return (
                                False,
                                validation_rules.get(
                                    "error_message",
                                    "Invalid date format for comparison",
                                ),
                                value,
                            )

                    # Return the normalized date value for storage
                    return True, "", parsed_value

                elif data_type == "string":
                    logger.info(f"Validating string field '{field_name}'")
                    # String validations
                    if (
                        "min_length" in validation_rules
                        and len(str(value)) < validation_rules["min_length"]
                    ):
                        logger.warning(
                            f"String length {len(str(value))} is less than minimum {validation_rules['min_length']}"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message",
                                f"Must be at least {validation_rules['min_length']} characters",
                            ),
                            value,
                        )
                    if (
                        "max_length" in validation_rules
                        and len(str(value)) > validation_rules["max_length"]
                    ):
                        logger.warning(
                            f"String length {len(str(value))} is greater than maximum {validation_rules['max_length']}"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message",
                                f"Must be no more than {validation_rules['max_length']} characters",
                            ),
                            value,
                        )
                    if "exact_value" in validation_rules and str(value) != str(
                        validation_rules["exact_value"]
                    ):
                        logger.warning(
                            f"String value '{value}' does not match exact value '{validation_rules['exact_value']}'"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message",
                                f"Must equal {validation_rules['exact_value']}",
                            ),
                            value,
                        )
                    if "not_equal_value" in validation_rules and str(
                        value
                    ) == str(validation_rules["not_equal_value"]):
                        logger.warning(
                            f"String value '{value}' matches not_equal_value '{validation_rules['not_equal_value']}'"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message",
                                f"Must not equal {validation_rules['not_equal_value']}",
                            ),
                            value,
                        )
                    if "regex_pattern" in validation_rules:
                        logger.info(
                            f"Checking regex pattern: {validation_rules['regex_pattern']}"
                        )
                        if not re.match(
                            validation_rules["regex_pattern"], str(value)
                        ):
                            logger.warning(
                                f"String value '{value}' does not match regex pattern"
                            )
                            return (
                                False,
                                validation_rules.get(
                                    "error_message",
                                    f"Must match pattern {validation_rules['regex_pattern']}",
                                ),
                                value,
                            )

                    return True, "", value
                elif data_type == "boolean":
                    logger.info(f"Validating boolean field '{field_name}'")
                    # Boolean validations
                    bool_value = None
                    if isinstance(value, bool):
                        bool_value = value
                    elif isinstance(value, str):
                        bool_value = value.lower() in ["true", "yes", "y", "1"]
                    logger.info(
                        f"Converted value '{value}' to boolean: {bool_value}"
                    )

                    if (
                        "exact_value" in validation_rules
                        and bool_value != validation_rules["exact_value"]
                    ):
                        expected = (
                            "Yes" if validation_rules["exact_value"] else "No"
                        )
                        logger.warning(
                            f"Boolean value {bool_value} does not match expected {expected}"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message", f"Must be {expected}"
                            ),
                            value,
                        )
                    if (
                        "not_equal_value" in validation_rules
                        and bool_value == validation_rules["not_equal_value"]
                    ):
                        not_expected = (
                            "Yes"
                            if validation_rules["not_equal_value"]
                            else "No"
                        )
                        logger.warning(
                            f"Boolean value {bool_value} matches not_expected {not_expected}"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message", f"Must not be {not_expected}"
                            ),
                            value,
                        )

                    return True, "", bool_value

                elif data_type == "number":
                    logger.info(f"Validating number field '{field_name}'")
                    try:
                        num_value = float(value)
                        logger.info(
                            f"Converted value '{value}' to number: {num_value}"
                        )

                        # Handle standard min/max value validation
                        if (
                            "min" in validation_rules
                            and num_value < validation_rules["min"]
                        ):
                            logger.warning(
                                f"Number {num_value} is less than minimum {validation_rules['min']}"
                            )
                            return (
                                False,
                                validation_rules.get(
                                    "error_message",
                                    f"Must be greater than {validation_rules['min']}",
                                ),
                                value,
                            )
                        if (
                            "max" in validation_rules
                            and num_value > validation_rules["max"]
                        ):
                            logger.warning(
                                f"Number {num_value} is greater than maximum {validation_rules['max']}"
                            )
                            return (
                                False,
                                validation_rules.get(
                                    "error_message",
                                    f"Must be less than {validation_rules['max']}",
                                ),
                                value,
                            )
                        if (
                            "exact_value" in validation_rules
                            and num_value
                            != float(validation_rules["exact_value"])
                        ):
                            logger.warning(
                                f"Number {num_value} does not match exact value {validation_rules['exact_value']}"
                            )
                            return (
                                False,
                                validation_rules.get(
                                    "error_message",
                                    f"Must equal {validation_rules['exact_value']}",
                                ),
                                value,
                            )
                        if (
                            "not_equal_value" in validation_rules
                            and num_value
                            == float(validation_rules["not_equal_value"])
                        ):
                            logger.warning(
                                f"Number {num_value} matches not_equal_value {validation_rules['not_equal_value']}"
                            )
                            return (
                                False,
                                validation_rules.get(
                                    "error_message",
                                    f"Must not equal {validation_rules['not_equal_value']}",
                                ),
                                value,
                            )

                        # Handle variable comparison validation
                        if (
                            "compare_variable" in validation_rules
                            and "compare_operator" in validation_rules
                        ):
                            compare_var_name = validation_rules[
                                "compare_variable"
                            ]
                            compare_operator = validation_rules[
                                "compare_operator"
                            ]
                            logger.info(
                                f"Comparing with variable '{compare_var_name}' using operator '{compare_operator}'"
                            )

                            # Get the comparison variable value from current variables
                            compare_value = current_variables.get(
                                compare_var_name
                            )
                            if compare_value is not None:
                                try:
                                    compare_num_value = float(compare_value)
                                    logger.info(
                                        f"Comparison value: {compare_num_value}"
                                    )

                                    if (
                                        compare_operator == "greater_than"
                                        and num_value <= compare_num_value
                                    ):
                                        error_msg = validation_rules.get(
                                            "error_message",
                                            f"{field_name} must be greater than {compare_var_name}",
                                        )
                                        logger.warning(
                                            f"Number {num_value} is not greater than {compare_num_value}"
                                        )
                                        return False, error_msg, value
                                    elif (
                                        compare_operator == "less_than"
                                        and num_value >= compare_num_value
                                    ):
                                        error_msg = validation_rules.get(
                                            "error_message",
                                            f"{field_name} must be less than {compare_var_name}",
                                        )
                                        logger.warning(
                                            f"Number {num_value} is not less than {compare_num_value}"
                                        )
                                        return False, error_msg, value
                                    elif (
                                        compare_operator == "equals"
                                        and num_value != compare_num_value
                                    ):
                                        error_msg = validation_rules.get(
                                            "error_message",
                                            f"{field_name} must equal {compare_var_name}",
                                        )
                                        logger.warning(
                                            f"Number {num_value} does not equal {compare_num_value}"
                                        )
                                        return False, error_msg, value
                                    elif (
                                        compare_operator == "not_equals"
                                        and num_value == compare_num_value
                                    ):
                                        error_msg = validation_rules.get(
                                            "error_message",
                                            f"{field_name} must not equal {compare_var_name}",
                                        )
                                        logger.warning(
                                            f"Number {num_value} equals {compare_num_value}"
                                        )
                                        return False, error_msg, value

                                except ValueError:
                                    logger.warning(
                                        f"Could not convert comparison variable {compare_var_name} value '{compare_value}' to number"
                                    )
                            else:
                                # If comparison variable doesn't exist yet, we might want to delay validation
                                logger.info(
                                    f"Comparison variable {compare_var_name} not set yet, skipping variable comparison for {field_name}"
                                )

                        # Return the converted number value for storage
                        return True, "", num_value

                    except ValueError:
                        logger.warning(
                            f"Could not convert value '{value}' to number"
                        )
                        return (
                            False,
                            validation_rules.get(
                                "error_message", "Value must be a number"
                            ),
                            value,
                        )

                logger.info(f"Validation passed for field '{field_name}'")
                return True, "", value

            def format_value_for_display(
                value: Any, field_name: str = None
            ) -> str:
                """Format values for display in a user-friendly way"""
                # Handle boolean values
                if isinstance(value, bool):
                    return "Yes" if value else "No"

                # Check if it's a string that represents a boolean
                if isinstance(value, str):
                    value_lower = value.lower()
                    if value_lower in ["yes", "true", "y", "1"]:
                        return "Yes"
                    if value_lower in ["no", "false", "n", "0"]:
                        return "No"

                # For all other values (dates, numbers, strings), return as-is
                return str(value)

            if action == "update_variable":
                if not variable_name:
                    return "❌ Error: variable_name is required for update_variable action"

                # Debug logging
                logger.info(
                    f"DEBUG: Updating variable '{variable_name}' with value: '{variable_value}' (type: {type(variable_value)})"
                )

                # Validate the variable value using baked-in validation
                is_valid, error_message, normalized_value = (
                    validate_variable_value(variable_name, variable_value)
                )
                if not is_valid:
                    return f"❌ Error: {error_message}"

                # Debug logging for normalized value
                logger.info(
                    f"DEBUG: After validation, normalized_value: '{normalized_value}' (type: {type(normalized_value)})"
                )

                # Update the variable
                current_variables[variable_name] = normalized_value

                # Update call metadata
                call_metadata["variables"] = current_variables
                AppConfig().set_call_metadata(call_metadata)

                # Debug logging for stored value
                logger.info(
                    f"DEBUG: Stored in AppConfig - current_variables: {current_variables}"
                )

                # Log the update
                logger.info(
                    f"Updated variable {variable_name} = {normalized_value}"
                )
                if notes:
                    logger.info(f"Notes: {notes}")

                status = calculate_completion_status(current_variables, config)

                return f"✅ **Updated {variable_name}** = {format_value_for_display(normalized_value)}\ncurrent status: {status}"

            elif action == "get_status":
                # Return current status of all variables
                status = calculate_completion_status(current_variables, config)

                result = "📊 **Variable Status Report**\n\n"
                result += f"**Required Fields ({status['required_fields']['completed']}/{status['required_fields']['total']}):**\n"

                for field in config.get("required_fields", []):
                    value = current_variables.get(field)
                    status_icon = (
                        "✅" if value is not None and value != "" else "❌"
                    )
                    description = config.get("field_descriptions", {}).get(
                        field, ""
                    )
                    formatted_value = (
                        format_value_for_display(value)
                        if value is not None
                        else "Not collected"
                    )
                    result += f"{status_icon} {field}: {formatted_value}\n"
                    if description:
                        result += f"   → {description}\n"

                if config.get("optional_fields"):
                    result += f"\n**Optional Fields ({status['optional_fields']['completed']}/{status['optional_fields']['total']}):**\n"
                    for field in config.get("optional_fields", []):
                        value = current_variables.get(field)
                        status_icon = (
                            "✅" if value is not None and value != "" else "⭕"
                        )
                        formatted_value = (
                            format_value_for_display(value)
                            if value is not None
                            else "Not collected"
                        )
                        result += f"{status_icon} {field}: {formatted_value}\n"

                result += f"\n**Overall Completion: {status['overall']['completion_rate']:.1%}**"
                if status["overall"]["can_complete"]:
                    result += "\n🎯 **Ready to complete call - all required fields collected!**"
                else:
                    result += (
                        "\n⚠️ **More information needed before completing call**"
                    )

                return result

            # Add other actions as needed...

            else:
                return f"❌ Error: Unknown action '{action}'"

        except Exception as e:
            logger.error(f"Error in manage_variables tool: {e}")
            return f"❌ Error managing variables: {str(e)}"

    return StructuredTool.from_function(
        func=manage_variables_impl,
        name="manage_variables",
        description="""
        Manage and track variables in the call state. Use this tool to:
        - Update variable values as you collect information
        - Check completion status and what's still needed

        This tool uses validation configuration specific to its instance/context.

        Actions (use EXACTLY as shown):
        - action: "get_status" - Get current status of all variables
        - action: "update_variable" - Update a specific variable value (requires variable_name and variable_value)

        Available variables:
        {available_variables}
        """.format(
            available_variables="\n".join(
                f"- {field}: {config.get('field_descriptions', {}).get(field, '')}"
                for field in config.get("required_fields", [])
                + config.get("optional_fields", [])
            )
        ),
        args_schema=VariableManagerRequest,
    )


@tool(args_schema=VariableManagerRequest)
def manage_variables(
    action: str,
    variable_name: Optional[str] = None,
    variable_value: Optional[Union[str, int, float, bool]] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Legacy manage_variables tool for backward compatibility.
    Use create_manage_variables_tool() factory for new implementations.

    This tool manages and tracks variables in the call state.

    Actions (use EXACTLY as shown):
    - action: "get_status" - Get current status of all variables
    - action: "update_variable" - Update a specific variable value (requires variable_name and variable_value)

    Args:
        action: The action to perform ("get_status" or "update_variable")
        variable_name: Name of the variable to update (required for update_variable action)
        variable_value: Value to set for the variable (required for update_variable action)
        notes: Optional notes about the update or context

    Returns:
        A string with the result of the action
    """
    # Create a tool with default config and call it
    # Note: create_manage_variables_tool will automatically use validation_config from call_config if available
    default_tool = create_manage_variables_tool()
    return default_tool.func(action, variable_name, variable_value, notes)
