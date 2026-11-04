import argparse
from contextlib import suppress

import spinedb_api
import scendesigner

def main():
    arg_parser = argparse.ArgumentParser(prog="create_scenario", description="Creates scenario for sensitivity analysis.")
    arg_parser.add_argument("db_url", help="Target Spine database URL.")
    args = arg_parser.parse_args()
    with spinedb_api.DatabaseMapping(args.db_url) as db_map:
        with spinedb_api.spine_db_client.lock_db(db_map):
            scendesigner.make_scenarios(db_map, db_map, db_map)
            with suppress(spinedb_api.exception.NothingToCommit):
                db_map.commit_session("Created scenario for sensitivity analysis.")


if __name__ == "__main__":
    main()
