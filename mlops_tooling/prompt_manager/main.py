import jinja2


class PromptManager:
    def __init__(self, path: str):
        self._path = path
        self._prompt = self._set_path()

    def _set_path(self):
        return jinja2.Environment(loader=jinja2.FileSystemLoader(self._path))

    def prompt(self, prompt_file: str, **kwargs) -> str:
        prompt_output = self._prompt.get_template(prompt_file).render(kwargs)
        return prompt_output
