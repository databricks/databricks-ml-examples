from jinja2 import Environment, PackageLoader, StrictUndefined, FileSystemLoader, BaseLoader


class TemplateManager:
    def __init__(self, loader: BaseLoader):
        self._env = Environment(loader=loader, trim_blocks=True, undefined=StrictUndefined)

    @staticmethod
    def get_package_loader(package: str):
        return PackageLoader(package, "templates")

    @staticmethod
    def get_filesystem_loader(location: str):
        return FileSystemLoader(location)

    def get_template(self, template_name):
        return self._env.get_template(template_name)

    def render_template(self, template_name, **kwargs):
        template = self.get_template(template_name)
        return template.render(**kwargs)

    def dump_template(self, template_name, outout_file, **kwargs):
        template = self.get_template(template_name)
        return template.stream(**kwargs).dump(outout_file)
