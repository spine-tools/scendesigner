import argparse
from contextlib import suppress

import spinedb_api
import scendesigner


def main():
    arg_parser = argparse.ArgumentParser(prog="create_config", description="Creates basic structures needed for sensitivity analysis.")
    arg_parser.add_argument("--parameters", nargs="*", help="Parameters to use.")
    arg_parser.add_argument("--classes", nargs="*", help="Classes to use.")
    arg_parser.add_argument("db_url", help="Target Spine database URL.")
    args = arg_parser.parse_args()
    with spinedb_api.DatabaseMapping(args.db_url) as db_map:
        scendesigner.make_structure(db_map)
        for parameter in args.parameters:
            scendesigner.add_parameter_to_analysis(db_map, parameter)
        for entity_class in args.classes:
            scendesigner.add_entity_class_to_analysis(db_map, entity_class)
        with suppress(spinedb_api.exception.NothingToCommit):
            db_map.commit_session("Created sensitivity analysis config.")


if __name__ == "__main__":
    main()
