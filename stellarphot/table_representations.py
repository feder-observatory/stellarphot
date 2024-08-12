from astropy.io.misc.yaml import AstropyDumper, AstropyLoader

from stellarphot.settings import models

__all__ = []


def generate_table_representers(cls):
    """
    Call this method during initialization of a class to add the YAML
    Table representation for the class.
    """
    class_string = f"!{cls.__name__}"

    # Add YAML round-tripping for the model
    def _representer(dumper, model):
        # THIS SHOULD TAP INTO ASTROPY'S YAML DUMPER SOMEHOW
        # This is a little hacky at the moment. It seems like YAML
        # has trouble reading in a dictionary, so though model.model_dump()
        # works fine for writing, we can't construct from the dump.
        #
        # Instead of figuring out the right way to do that, this just dumps
        # the json representation as a string.
        return dumper.represent_mapping(
            class_string, {"model_json_string": model.model_dump_json()}
        )

    def _constructor(loader, node):
        # This loads the simple dictionary we dumped in _representer,
        # then initializes the model with the json string.
        mapping = loader.construct_mapping(node)
        return cls.model_validate_json(mapping["model_json_string"])

    AstropyDumper.add_representer(cls, _representer)
    AstropyLoader.add_constructor(class_string, _constructor)


# This code is deliberately executable so that it can be executed on import, which
# should assure that Table representations are generated for all models.
for model_name in models.__all__:
    model_class = getattr(models, model_name)
    generate_table_representers(model_class)
