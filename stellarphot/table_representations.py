import json

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


def serialize_models_in_table_meta(table_meta):
    """
    Serialize the models in the table metadata **IN PLACE**.
    This is used to ensure that the models are represented as simple
    dictionaries when written to disk.

    Parameters
    ----------
    table_meta : dict
        The metadata dictionary of the table.
    """
    model_classes = tuple(getattr(models, model_name) for model_name in models.__all__)

    for key, value in table_meta.items():
        # If the value is a model instance, serialize it
        if isinstance(value, model_classes):
            model_instance = value
            # So, funny story. model_dump gives you a nice dictionary, in which
            # things like Longitude are turned into strings. However, writing them
            # to ECSV fails, because ECSV doesn't understand np.str_, and - guess what -
            # a Longitude returns a np.str_ when you do str(some_longitude).
            # The upshot is that the workaround here, i.e. using model_dump to get
            # simple objects into the header, does not work unless all string values
            # are converted to str.
            #
            # The issue has been reported in
            # https://github.com/astropy/astropy/issues/18235
            #

            # Dumping to json ensures that all the objects are converted to
            # very basic types, which is easy enough to convert to a dictionary.

            # Use model_dump_json to get a simple dictionary representation
            model_json = model_instance.model_dump_json()
            model_dict = json.loads(model_json)
            table_meta[key] = model_dict
            table_meta[key]["_model_name"] = model_instance.__class__.__name__
        # If the value is a dict, recurse
        elif isinstance(value, dict):
            serialize_models_in_table_meta(value)


def deserialize_models_in_table_meta(table_meta):
    """
    Deserialize the models in the table metadata **IN PLACE**.
    This is used to ensure that the models are restored from simple
    dictionaries when read from disk.

    There are two places a model might be stored:

    1. Directly in the table metadata.
    2. As a TableAttribute, which ends up in the table's meta under
       the "``__attributes__``" key.

    This function checks subdictionaries recursively to properly handle this
    case.

    Parameters
    ----------
    table_meta : dict
        The metadata dictionary of the table.
    """
    known_models = {
        model_name: getattr(models, model_name) for model_name in models.__all__
    }

    model_keys_in_meta = []
    for key, value in table_meta.items():
        # Check if the value is a dictionary and has a "_model_name" key
        if isinstance(value, dict):
            if "_model_name" in value:
                # Check if the model name is in the known models
                if value["_model_name"] in known_models.keys():
                    model_keys_in_meta.append(key)
            else:
                # Time to recurse into the dictionary
                deserialize_models_in_table_meta(value)

    for key in model_keys_in_meta:
        model_name = table_meta[key].pop("_model_name")
        table_meta[key] = known_models[model_name].model_validate(table_meta[key])


def _generate_old_table_representers():
    """
    This provides what is needed to read the "old-style" data tables in
    which the models were stored as objects in the table metadata.
    """
    for model_name in models.__all__:
        model_class = getattr(models, model_name)
        generate_table_representers(model_class)
