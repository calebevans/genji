<div align="center">

# Genji

[![PyPI version](https://img.shields.io/pypi/v/genji.svg)](https://pypi.org/project/genji/)
[![Python versions](https://img.shields.io/pypi/pyversions/genji.svg)](https://pypi.org/project/genji/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

Genji is a templating library for LLM-generated structured output, built on [Jinja2](https://jinja.palletsprojects.com/). It ensures templates own the structure and syntax (JSON brackets, HTML tags, YAML indentation) while LLMs only generate content, guaranteeing valid output every time.

**The problem:** LLMs often produce malformed JSON, broken HTML, or invalid YAML when asked to generate structured output directly.

**The solution:** Separate concerns. Templates define structure, LLMs fill in content. Structure is guaranteed, content is generated.

## Installation

### From PyPI (Recommended)

```bash
# With uv (recommended)
uv pip install genji

# With pip
pip install genji
```

### From Source

```bash
git clone https://github.com/calebevans/genji.git
cd genji
uv pip install -e .
```

For development:

```bash
uv pip install -e ".[dev]"
pre-commit install
```

## Quick Start

```python
from genji import Template, LLMBackend

# Configure the LLM backend
backend = LLMBackend(model="gpt-4o-mini")

# Define a template (default_filter="json" applies to all gen() calls)
template = Template("""
{
  "greeting": {{ gen("a friendly greeting for {name}") }},
  "farewell": {{ gen("a warm farewell for {name}") }}
}
""", backend=backend, default_filter="json")

# Render with variables
result = template.render(name="Alice")
print(result)  # Valid JSON guaranteed

# Or parse directly to dict
data = template.render_json(name="Alice")
print(data["greeting"])  # LLM-generated greeting
```

> **Note:** On first run, LiteLLM may download model configurations. Subsequent runs use cached data.

## Features

### Template Syntax

Genji extends Jinja2 with a `gen()` function for LLM generation:

```python
# Basic generation
{{ gen("a creative tagline") }}

# With variable interpolation
{{ gen("a description of {product}") }}

# With generation parameters
{{ gen("a tweet", max_tokens=280, temperature=0.9) }}

# With filters for different formats
{{ gen("content") | json }}   # JSON-safe string with quotes
{{ gen("content") | html }}   # HTML entity escaping
{{ gen("content") | yaml }}   # YAML-safe string
{{ gen("content") | xml }}    # XML entity escaping
```

All standard Jinja2 features are supported:

```jinja2
{# Comments #}

{% if condition %}
  {{ gen("something") }}
{% endif %}

{% for item in items %}
  {{ gen("content for {item}") }}
{% endfor %}
```

### Format-Specific Filters

Genji provides filters for safe escaping in different formats:

| Filter | Purpose | Example Output |
|--------|---------|----------------|
| `json` | JSON string with quotes | `"Hello \"World\""` |
| `html` | HTML entity escaping | `&lt;b&gt;text&lt;/b&gt;` |
| `xml` | XML entity escaping | `&lt;tag&gt;content&lt;/tag&gt;` |
| `yaml` | YAML-safe string | `"key: value"` |
| `raw` | No escaping (use carefully!) | `<dangerous>` |
| `strip` | Remove whitespace | `"text"` |
| `lower` | Lowercase | `"hello"` |
| `upper` | Uppercase | `"HELLO"` |
| `truncate(n)` | Truncate to n chars | `"Long te..."` |

**Important:** The `json` filter outputs a complete JSON string value including quotes:
```python
{{ gen("text") | json }}  # Outputs: "the generated text"
```

### Default Filters

Avoid repetition by setting a default filter:

```python
# Apply | json to all gen() calls automatically
template = Template(source, backend, default_filter="json")

# Or use file extension auto-detection
template = Template.from_file("report.json.genji", backend)
# Auto-detects "json" filter from .json.genji extension

# Override for specific prompts when needed
{{ gen("normal content") }}      # Gets json filter
{{ gen("special") | raw }}       # Skips filter
{{ gen("html content") | html }} # Uses html instead
```

### LLM Backend Support

Genji uses [LiteLLM](https://github.com/BerriAI/litellm) for unified access to 100+ LLM providers.

See the [full list of supported models](https://models.litellm.ai/).

#### OpenAI

```python
backend = LLMBackend(
    model="gpt-4o-mini",
    api_key="sk-...",  # or set OPENAI_API_KEY env var
)
```

#### Anthropic Claude

```python
backend = LLMBackend(
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
)
```

#### Google Gemini

```python
backend = LLMBackend(
    model="gemini/gemini-2.5-flash",
    api_key="...",  # or set GEMINI_API_KEY env var
)
```

#### Local Ollama

```python
backend = LLMBackend(
    model="ollama/llama3",
    base_url="http://localhost:11434"
)
```

#### Azure OpenAI

```python
backend = LLMBackend(
    model="azure/your-deployment-name",
    api_key="...",
    base_url="https://your-resource.openai.azure.com"
)
```

For a complete list of supported models, see [LiteLLM's model documentation](https://models.litellm.ai/).

### Loading Templates from Files

```python
# Create a template file: templates/report.json.genji
template = Template.from_file("templates/report.json.genji", backend)
result = template.render(topic="climate change")
```

### Batch Generation

Genji automatically batches multiple `gen()` calls for efficiency:

```python
template = Template("""
{
  "field1": {{ gen("prompt1") | json }},
  "field2": {{ gen("prompt2") | json }},
  "field3": {{ gen("prompt3") | json }}
}
""", backend=backend)

# All 3 prompts are sent to the LLM in parallel!
result = template.render()
```

## API Reference

### Template

```python
class Template:
    def __init__(
        self,
        source: str,
        backend: LLMBackend | MockBackend,
        default_filter: str | None = None
    ) -> None:
        """Initialize a template from a string.

        Args:
            source: Template string with Jinja2 syntax and gen() calls.
            backend: LLM backend instance (LLMBackend or MockBackend).
            default_filter: Optional default filter to apply to all gen() calls
                (e.g., "json", "html", "yaml"). Can be overridden per-prompt.
        """

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        backend: LLMBackend | MockBackend,
        default_filter: str | None = None
    ) -> Template:
        """Load a template from a file.

        Args:
            path: Path to template file.
            backend: LLM backend instance.
            default_filter: Optional default filter. If None, auto-detects from
                file extension (.json.genji -> "json", .html.genji -> "html", etc.).
        """

    def render(self, **context: Any) -> str:
        """Render the template with the given context variables.

        Returns:
            Rendered template as a string.
        """

    def render_json(self, **context: Any) -> dict[str, Any]:
        """Render the template and parse as JSON.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            TemplateRenderError: If output is not valid JSON.
        """
```

### LLMBackend

```python
class LLMBackend:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        add_system_prompt: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the LiteLLM backend.

        Args:
            model: Model name (defaults to "gpt-4o-mini").
            api_key: API key (or set via environment variable).
            base_url: Base URL for custom endpoints.
            temperature: Temperature for generation (None uses provider default).
            max_tokens: Max tokens per generation (None uses provider default).
            add_system_prompt: Whether to add instruction for concise responses.
                Defaults to True.
            **kwargs: Additional arguments passed to litellm.completion().
        """
```

#### Per-Prompt Parameters

You can configure generation parameters for individual `gen()` calls:

```python
# Control tokens, temperature, and stop sequences per prompt
{{ gen("short title", max_tokens=20) }}
{{ gen("creative content", temperature=0.9) }}
{{ gen("haiku", stop=["\n\n"]) }}
```

#### Smart Prompting

By default, Genji adds a system instruction to ensure LLMs return literal, concise responses:

```python
# Default - LLM returns exactly what's requested
backend = LLMBackend(model="gpt-4o-mini")
# "a title" returns one title, not a list of options

# Disable for full control
backend = LLMBackend(model="gpt-4o-mini", add_system_prompt=False)
```

### MockBackend

For testing without API calls:

```python
from genji import MockBackend

backend = MockBackend(default_response="Test content")
# or
backend = MockBackend(response_fn=lambda prompt: f"Response to: {prompt}")
```

## Configuration

| Parameter | Default | Environment Variable | Description |
|-----------|---------|---------------------|-------------|
| `model` | `gpt-4o-mini` | `GENJI_MODEL` | LLM model name |
| `api_key` | None | `GENJI_API_KEY` | API key for provider |
| `base_url` | None | `GENJI_BASE_URL` | Custom endpoint URL |
| `temperature` | Provider default | N/A | Temperature for generation |
| `max_tokens` | Provider default | N/A | Max tokens per generation |
| `add_system_prompt` | `True` | N/A | Add conciseness instruction |

## Error Handling

Genji provides clear exception types:

- `GenjiError` - Base exception
- `TemplateParseError` - Invalid template syntax
- `TemplateRenderError` - Error during rendering
- `BackendError` - LLM backend failure
- `FilterError` - Filter application failure
