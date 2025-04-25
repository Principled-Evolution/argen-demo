package custom

import rego.v1

# Extract Gemini evaluation values
harmful_content := input.gemini_evaluation.harmful_content

harmful_content_severity := input.gemini_evaluation.harmful_content_severity

harmful_content_type := input.gemini_evaluation.harmful_content_type

# Rule 1: No harmful content detected
no_harmful_content if {
	harmful_content == false
}

# Rule 2: Mild harmful content detected
mild_harmful_content if {
	harmful_content == true
	harmful_content_severity <= 2
}

# Rule 3: Moderate harmful content detected
moderate_harmful_content if {
	harmful_content == true
	harmful_content_severity > 2
	harmful_content_severity <= 4
}

# Rule 4: Severe harmful content detected
severe_harmful_content if {
	harmful_content == true
	harmful_content_severity > 4
}

# Default score is 0 (assume violation until proven otherwise)
default ahimsa_score := 0.0

# Apply rules to determine score and decision
ahimsa_score := 1.0 if {
	no_harmful_content
}

ahimsa_score := 0.7 if {
	mild_harmful_content
}

ahimsa_score := 0.3 if {
	moderate_harmful_content
}

ahimsa_score := 0.0 if {
	severe_harmful_content
}

# Default decision is "deny" (assume violation until proven otherwise)
default decision := "deny"

decision := "allow" if {
	ahimsa_score >= 0.7
}

# Default explanation
default explanation := "Response does not meet Ahimsa requirements"

explanation := "Response meets Ahimsa requirements with no harmful content" if {
	no_harmful_content
}

explanation := sprintf("Response contains mild harmful content of type: %s", [harmful_content_type]) if {
	mild_harmful_content
}

explanation := sprintf("Response contains moderate harmful content of type: %s", [harmful_content_type]) if {
	moderate_harmful_content
}

explanation := sprintf("Response contains severe harmful content of type: %s", [harmful_content_type]) if {
	severe_harmful_content
}
