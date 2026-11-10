# Copyright scendesigner contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, replace
import logging
from typing import TypeAlias, ClassVar

import numpy as np
import pandas as pd
import pydoe
import spinedb_api


class ScendesignerException(Exception):
    pass


@dataclass(frozen=True)
class FullFactorialMethod:
    NAME: ClassVar[str] = "full factorial"
    n_levels: int | list[int]


@dataclass(frozen=True)
class MorrisMethod:
    NAME: ClassVar[str] = "Morris method"
    n_trajectories: int
    n_levels: int  # must be even
    random_seed: int


Atoms: TypeAlias = tuple[str, ...]
Path: TypeAlias = tuple[str, Atoms, str, str]
Method: TypeAlias = FullFactorialMethod | MorrisMethod


ALL_METHODS: list[type[Method]] = [FullFactorialMethod, MorrisMethod]


def display_byname(byname: str | Atoms) -> str:
    if isinstance(byname, str):
        return byname
    return " | ".join(byname)


def make_structure(config_db_map: spinedb_api.DatabaseMapping) -> None:
    config_db_map.add_or_update_parameter_value_list(name="SensitivityAnalysisMethod")
    for i, method in enumerate(sorted(ALL_METHODS, key=lambda m: m.NAME)):
        config_db_map.add_or_update_list_value(
            parameter_value_list_name="SensitivityAnalysisMethod",
            parsed_value=method.NAME,
            index=i,
        )
    config_db_map.add_or_update_entity_class(
        name="SensitivityAnalysis",
        description="The first entity found in this class bears settings for sensitivity analysis.",
    )
    config_db_map.add_or_update_parameter_definition(
        entity_class_name="SensitivityAnalysis",
        name="version",
        parsed_value=1,
        parameter_type_list=("float",),
        description="Don't modify. Sensitivity analysis schema version.",
    )
    config_db_map.add_or_update_parameter_definition(
        entity_class_name="SensitivityAnalysis",
        name="method",
        parameter_value_list_name="SensitivityAnalysisMethod",
        parsed_value=MorrisMethod.NAME,
        description="Method used for sensitivity analysis.",
    )
    config_db_map.add_or_update_parameter_definition(
        entity_class_name="SensitivityAnalysis",
        name="random_seed",
        parsed_value=1337,
        parameter_type_list=("float",),
        description="Random seed used for sensitivity analysis.",
    )
    config_db_map.add_or_update_parameter_definition(
        entity_class_name="SensitivityAnalysis",
        name="trajectories",
        parsed_value=10,
        parameter_type_list=("float",),
        description="The number of trajectories in Morris method.",
    )
    config_db_map.add_or_update_parameter_definition(
        entity_class_name="SensitivityAnalysis",
        name="levels",
        parsed_value=4,
        parameter_type_list=("float",),
        description="The number of distinct values across all parameters, e.g. 4 levels for [0, 1] interval means 0, 1/3, 2/3 and 1.0. Usually even.",
    )
    config_db_map.add_or_update_entity_class(
        name="SensitiveParameter",
        description="The entities in this class are the parameters used in sensitivity analysis.",
    )


def set_method(config_db_map: spinedb_api.DatabaseMapping, method: Method) -> None:
    parameter_definition_table = config_db_map.mapped_table("parameter_definition")
    config_db_map.update(
        parameter_definition_table,
        entity_class_name="SensitivityAnalysis",
        name="method",
        parsed_value=method.NAME,
    )
    match method:
        case FullFactorialMethod():
            config_db_map.update(
                parameter_definition_table,
                entity_class_name="SensitivityAnalysis",
                name="levels",
                parsed_value=method.n_levels,
            )
        case MorrisMethod():
            config_db_map.update(
                parameter_definition_table,
                entity_class_name="SensitivityAnalysis",
                name="trajectories",
                parsed_value=method.n_trajectories,
            )
            config_db_map.update(
                parameter_definition_table,
                entity_class_name="SensitivityAnalysis",
                name="levels",
                parsed_value=method.n_levels,
            )
            config_db_map.update(
                parameter_definition_table,
                entity_class_name="SensitivityAnalysis",
                name="random_seed",
                parsed_value=method.random_seed,
            )
        case _:
            raise ValueError("unknown method type")


def config_class_name(entity_class_name: str) -> str:
    return f"{entity_class_name}__SensitiveParameter"


def add_entity_class_to_analysis(
    config_db_map: spinedb_api.DatabaseMapping, entity_class_name: str
) -> None:
    name = config_class_name(entity_class_name)
    parameter_class_dimensions = [entity_class_name, "SensitiveParameter"]
    parameter_class = config_db_map.add_or_update_entity_class(
        name=name,
        dimension_name_list=parameter_class_dimensions,
        description="A relationship linking entities to parameters used in sensitivity analysis.",
    )
    if parameter_class is None:
        return
    class_id = parameter_class["id"]
    config_db_map.add_parameter_definition(
        entity_class_id=class_id,
        name="variability_fraction",
        parameter_type_list=("float",),
        description="Vary x within [x - variability_fraction * x...x + variability_fraction * x].",
    )
    config_db_map.add_parameter_definition(
        entity_class_id=class_id,
        name="negative_limit",
        parameter_type_list=("float",),
        description="Vary x within [x + negative_limit...x + positive_limit].",
    )
    config_db_map.add_parameter_definition(
        entity_class_id=class_id,
        name="positive_limit",
        parameter_type_list=("float",),
        description="Vary x within [x + negative_limit...x + positive_limit].",
    )
    config_db_map.add_parameter_definition(
        entity_class_id=class_id,
        name="levels",
        parameter_type_list=("float",),
        description="The number of distinct values, e.g. 4 levels for [0, 1] means 0, 0.33, 0.67 and 1. Must be even with Morris method.",
    )


def add_parameter_to_analysis(
    config_db_map: spinedb_api.DatabaseMapping, parameter_definition_name: str
) -> None:
    config_db_map.add_or_update_entity(
        entity_class_name="SensitiveParameter", name=parameter_definition_name
    )


def add_entity_to_analysis(
    config_db_map: spinedb_api.DatabaseMapping,
    entity_class_name: str,
    entity_byname: Atoms,
    parameter_definition_name: str,
) -> None:
    analysis_class = config_class_name(entity_class_name)
    analysis_byname = entity_byname + (parameter_definition_name,)
    config_db_map.add_or_update_entity(
        entity_class_name=analysis_class, entity_byname=analysis_byname
    )


def add_variability_fraction(
    config_db_map: spinedb_api.DatabaseMapping,
    entity_class_name: str,
    entity_byname: Atoms,
    parameter_definition_name: str,
    alternative_name: str,
    variability_fraction: float,
) -> None:
    analysis_class = config_class_name(entity_class_name)
    analysis_byname = entity_byname + (parameter_definition_name,)
    config_db_map.add_or_update_parameter_value(
        entity_class_name=analysis_class,
        entity_byname=analysis_byname,
        parameter_definition_name="variability_fraction",
        alternative_name=alternative_name,
        parsed_value=variability_fraction,
    )


def add_limits(
    config_db_map: spinedb_api.DatabaseMapping,
    entity_class_name: str,
    entity_byname: Atoms,
    parameter_definition_name: str,
    alternative_name: str,
    negative_limit: float,
    positive_limit: float,
) -> None:
    analysis_class = config_class_name(entity_class_name)
    analysis_byname = entity_byname + (parameter_definition_name,)
    config_db_map.add_or_update_parameter_value(
        entity_class_name=analysis_class,
        entity_byname=analysis_byname,
        parameter_definition_name="negative_limit",
        alternative_name=alternative_name,
        parsed_value=negative_limit,
    )
    config_db_map.add_or_update_parameter_value(
        entity_class_name=analysis_class,
        entity_byname=analysis_byname,
        parameter_definition_name="positive_limit",
        alternative_name=alternative_name,
        parsed_value=positive_limit,
    )


def add_levels(
    config_db_map: spinedb_api.DatabaseMapping,
    entity_class_name: str,
    entity_byname: Atoms,
    parameter_definition_name: str,
    alternative_name: str,
    levels: int,
) -> None:
    analysis_class = config_class_name(entity_class_name)
    analysis_byname = entity_byname + (parameter_definition_name,)
    config_db_map.add_or_update_parameter_value(
        entity_class_name=analysis_class,
        entity_byname=analysis_byname,
        parameter_definition_name="levels",
        alternative_name=alternative_name,
        parsed_value=float(levels),
    )


@dataclass
class Fractional:
    value: float


@dataclass
class Range:
    negative_limit: float
    positive_limit: float


Variability: TypeAlias = Fractional | Range


@dataclass
class ParameterAnalysisData:
    parameter_definition_name: str
    variability: Variability


@dataclass
class EntityAnalysisData:
    entity_byname: Atoms
    parameters: list[ParameterAnalysisData]


@dataclass
class EntityClassAnalysisData:
    entity_class_name: str
    entities: list[EntityAnalysisData]


def add_variability(
    config_db_map: spinedb_api.DatabaseMapping,
    entity_class_name: str,
    entity_byname: Atoms,
    parameter_definition_name: str,
    alternative_name: str,
    variability: Variability,
) -> None:
    match variability:
        case Fractional():
            add_variability_fraction(
                config_db_map,
                entity_class_name,
                entity_byname,
                parameter_definition_name,
                alternative_name,
                variability.value,
            )
        case Range():
            add_limits(
                config_db_map,
                entity_class_name,
                entity_byname,
                parameter_definition_name,
                alternative_name,
                variability.negative_limit,
                variability.positive_limit,
            )
        case _:
            raise ValueError("unknown variability type")


def build_analysis(
    config_db_map: spinedb_api.DatabaseMapping,
    in_db_map: spinedb_api.DatabaseMapping,
    base_scenario: str,
    data: list[EntityClassAnalysisData],
) -> None:
    alternatives = in_db_map.scenario(name=base_scenario)["alternative_name_list"]
    value_table = in_db_map.mapped_table("parameter_value")
    for class_data in data:
        add_entity_class_to_analysis(config_db_map, class_data.entity_class_name)
        for entity_data in class_data.entities:
            for parameter_data in entity_data.parameters:
                add_parameter_to_analysis(
                    config_db_map, parameter_data.parameter_definition_name
                )
                add_entity_to_analysis(
                    config_db_map,
                    class_data.entity_class_name,
                    entity_data.entity_byname,
                    parameter_data.parameter_definition_name,
                )
                for alternative in reversed(alternatives):
                    try:
                        in_db_map.item(
                            value_table,
                            entity_class_name=class_data.entity_class_name,
                            entity_byname=entity_data.entity_byname,
                            parameter_definition_name=parameter_data.parameter_definition_name,
                            alternative_name=alternative,
                        )
                    except spinedb_api.SpineDBAPIError:
                        continue
                    else:
                        add_variability(
                            config_db_map,
                            class_data.entity_class_name,
                            entity_data.entity_byname,
                            parameter_data.parameter_definition_name,
                            alternative,
                            parameter_data.variability,
                        )
                        break
                else:
                    logging.warning(
                        f"no value for {parameter_data.parameter_definition_name} (entity {entity_data.entity_byname}, class {class_data.entity_class_name}) in scenario {base_scenario}"
                    )


def read_method(config_db_map: spinedb_api.DatabaseMapping) -> Method:
    analysis_class = config_db_map.entity_class(name="SensitivityAnalysis")
    default_method = config_db_map.parameter_definition(
        entity_class_id=analysis_class["id"], name="method"
    )["parsed_value"]
    if default_method == MorrisMethod.NAME:
        default_trajectories = int(
            config_db_map.parameter_definition(
                entity_class_id=analysis_class["id"], name="trajectories"
            )["parsed_value"]
        )
        default_levels = int(
            config_db_map.parameter_definition(
                entity_class_id=analysis_class["id"], name="levels"
            )["parsed_value"]
        )
        default_seed = int(
            config_db_map.parameter_definition(
                entity_class_id=analysis_class["id"], name="random_seed"
            )["parsed_value"]
        )
        return MorrisMethod(default_trajectories, default_levels, default_seed)
    if default_method == FullFactorialMethod.NAME:
        default_levels = int(
            config_db_map.parameter_definition(
                entity_class_id=analysis_class["id"], name="levels"
            )["parsed_value"]
        )
        return FullFactorialMethod(default_levels)
    raise RuntimeError(f"unknown analysis method {default_method}")


SamplingArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def generate_full_factorial_sampling(
    method: FullFactorialMethod, n_parameters: int = -1
) -> SamplingArray:
    levels = (
        n_parameters * [method.n_levels]
        if isinstance(method.n_levels, int)
        else method.n_levels
    )
    return pydoe.fullfact(levels) / (np.array(levels) - 1)


def generate_morris_sampling(method: MorrisMethod, n_parameters: int) -> SamplingArray:
    return pydoe.morris_sampling(
        n_parameters, method.n_trajectories, method.n_levels, method.random_seed
    )


def generate_sampling(method, n_parameters: int) -> SamplingArray:
    match method:
        case FullFactorialMethod():
            return generate_full_factorial_sampling(method, n_parameters)
        case MorrisMethod():
            return generate_morris_sampling(method, n_parameters)
    raise RuntimeError(f"unknown sampling method {method}")


def apply_sampling(
    analysis_parameters: pd.DataFrame, sampling: SamplingArray
) -> pd.DataFrame:
    negative_limits = analysis_parameters.iloc[:, 0]
    positive_limits = analysis_parameters.iloc[:, 1]
    variabilities = analysis_parameters.iloc[:, 2]
    values = analysis_parameters.iloc[:, 3]
    ranges = (positive_limits - negative_limits) / 2
    ranges = ranges.combine_first(variabilities * values / 2)
    center_adjustments = (positive_limits + negative_limits) / 2
    center_adjustments = center_adjustments.fillna(0.0)
    varied_values = []
    for scenario_number in range(len(sampling)):
        varied = (
            values
            + center_adjustments
            + (2 * sampling[scenario_number, :] - 1) * ranges
        )
        varied_values.append(varied)
    return pd.concat(varied_values, axis=1)


ConfigValues: TypeAlias = dict[str, spinedb_api.db_mapping_base.PublicItem]


@dataclass(frozen=True)
class RangeConfigs:
    negative_limits: ConfigValues
    positive_limits: ConfigValues
    variability_fractions: ConfigValues


def read_config_values_by_alternative(
    config_db_map: spinedb_api.DatabaseMapping,
    parameter_value_table: spinedb_api.db_mapping_base.MappedTable,
    config_entity: spinedb_api.db_mapping_base.PublicItem,
) -> RangeConfigs:
    negative_limits = {}
    positive_limits = {}
    variability_fractions = {}
    config_entity_id = config_entity["id"]
    for config_value in config_db_map.find(
        parameter_value_table,
        entity_id=config_entity_id,
    ):
        alternative = config_value["alternative_name"]
        match config_value["parameter_definition_name"]:
            case "negative_limit":
                negative_limits[alternative] = config_value
            case "positive_limit":
                positive_limits[alternative] = config_value
            case "variability_fraction":
                variability_fractions[alternative] = config_value
    return RangeConfigs(negative_limits, positive_limits, variability_fractions)


OptionalInt: TypeAlias = int | pd.api.typing.NAType


def read_individual_levels_for_scenario(
    config_db_map: spinedb_api.DatabaseMapping,
    alternatives: list[str],
    range_config_index: pd.MultiIndex,
) -> list[OptionalInt]:
    levels = []
    for i in range_config_index:
        in_class, in_entity_byname, in_parameter = i
        config_class = config_class_name(in_class)
        config_byname = in_entity_byname + (in_parameter,)
        for alternative in reversed(alternatives):
            try:
                levels_for_alternative = config_db_map.parameter_value(
                    entity_class_name=config_class,
                    entity_byname=config_byname,
                    parameter_definition_name="levels",
                    alternative_name=alternative,
                )["parsed_value"]
            except spinedb_api.SpineDBAPIError:
                continue
            else:
                levels.append(int(levels_for_alternative))
                break
        else:
            levels.append(pd.NA)
    return levels


OptionalFloat: TypeAlias = float | pd.api.typing.NAType


def config_values_by_scenario(
    alternatives: list[str], range_configs: RangeConfigs
) -> tuple[str, OptionalFloat, OptionalFloat, OptionalFloat] | None:
    negative_limit = pd.NA
    positive_limit = pd.NA
    variability_fraction = pd.NA
    for alternative in reversed(alternatives):
        if alternative in range_configs.positive_limits:
            positive_limit = range_configs.positive_limits[alternative]["parsed_value"]
            if not pd.isna(negative_limit):
                variability_fraction = pd.NA
                break
        if alternative in range_configs.negative_limits:
            negative_limit = range_configs.negative_limits[alternative]["parsed_value"]
            if not pd.isna(positive_limit):
                variability_fraction = pd.NA
                break
        elif alternative in range_configs.variability_fractions:
            variability_fraction = range_configs.variability_fractions[alternative][
                "parsed_value"
            ]
            break
    else:
        return None
    return alternative, negative_limit, positive_limit, variability_fraction


def read_analysis_parameters_for_scenario(
    config_db_map: spinedb_api.DatabaseMapping,
    in_db_map: spinedb_api.DatabaseMapping,
    alternatives: list[str],
) -> pd.DataFrame:
    indices: list[tuple[str, Atoms, str]] = []
    positive_limits: list[OptionalFloat] = []
    negative_limits: list[OptionalFloat] = []
    variability_fractions: list[OptionalFloat] = []
    in_values: list[float] = []
    config_entity_class_table = config_db_map.mapped_table("entity_class")
    config_entity_table = config_db_map.mapped_table("entity")
    config_parameter_value_table = config_db_map.mapped_table("parameter_value")
    in_parameter_value_table = in_db_map.mapped_table("parameter_value")
    for config_class in config_db_map.find(
        config_entity_class_table,
        dimension_name_list=[spinedb_api.Asterisk, "SensitiveParameter"],
    ):
        in_class_name = config_class["dimension_name_list"][0]
        class_name = config_class["name"]
        for config_entity in config_db_map.find(
            config_entity_table, entity_class_name=class_name
        ):
            range_configs = read_config_values_by_alternative(
                config_db_map, config_parameter_value_table, config_entity
            )
            config_values = config_values_by_scenario(alternatives, range_configs)
            if config_values is None:
                continue
            alternative, negative_limit, positive_limit, variability_fraction = (
                config_values
            )
            negative_limits.append(negative_limit)
            positive_limits.append(positive_limit)
            variability_fractions.append(variability_fraction)
            config_entity_byname = config_entity["entity_byname"]
            entity_byname = config_entity_byname[:-1]
            parameter_definition_name = config_entity_byname[-1]
            indices.append((in_class_name, entity_byname, parameter_definition_name))
            in_value = in_db_map.item(
                in_parameter_value_table,
                entity_class_name=in_class_name,
                entity_byname=entity_byname,
                parameter_definition_name=parameter_definition_name,
                alternative_name=alternative,
            )["parsed_value"]
            in_values.append(in_value)
    index = pd.MultiIndex.from_tuples(
        indices, names=["entity_class", "entity", "parameter_definition"]
    )
    return pd.DataFrame(
        {
            "negative_limit": negative_limits,
            "positive_limit": positive_limits,
            "variability_fraction": variability_fractions,
            "value": in_values,
        },
        index=index,
    ).sort_index()


def out_alternative_name(name_prefix: str, serial_number: int) -> str:
    return f"{name_prefix}_{serial_number + 1:03}"


def write_out_alternatives(
    out_db_map: spinedb_api.DatabaseMapping, name_prefix: str, count: int
) -> None:
    alternatives: list[str] = []
    for i in range(count):
        alternative_name = out_alternative_name(name_prefix, i)
        alternatives.append(alternative_name)
        out_db_map.add_alternative(
            name=alternative_name, description="For sensitivity analysis."
        )


def write_samples(
    out_db_map: spinedb_api.DatabaseMapping,
    alternative_name_prefix: str,
    samples: pd.DataFrame,
) -> None:
    for row_tuple in samples.itertuples(name=None):
        entity_class, entity_byname, parameter_definition = row_tuple[0]
        for alternative_i, value in enumerate(row_tuple[1:]):
            alternative_name = out_alternative_name(
                alternative_name_prefix, alternative_i
            )
            out_db_map.add_parameter_value(
                entity_class_name=entity_class,
                entity_byname=entity_byname,
                parameter_definition_name=parameter_definition,
                alternative_name=alternative_name,
                parsed_value=value,
            )


def write_scenarios(
    out_db_map: spinedb_api.DatabaseMapping,
    scenario_name_prefix: str,
    scenario_count: int,
    base_alternatives: list[str],
    alternative_name_prefix: str,
) -> None:
    for i in range(scenario_count):
        scenario_name = f"{scenario_name_prefix}_{i + 1:03}"
        out_db_map.add_scenario(
            name=scenario_name, description="For sensitivity analysis."
        )
        for rank, alternative in enumerate(base_alternatives):
            out_db_map.add_scenario_alternative(
                scenario_name=scenario_name, alternative_name=alternative, rank=rank
            )
        alternative_name = out_alternative_name(alternative_name_prefix, i)
        out_db_map.add_scenario_alternative(
            scenario_name=scenario_name,
            alternative_name=alternative_name,
            rank=len(base_alternatives),
        )


def make_scenarios(
    config_db_map: spinedb_api.DatabaseMapping,
    in_db_map: spinedb_api.DatabaseMapping,
    out_db_map: spinedb_api.DatabaseMapping,
) -> None:
    method = read_method(config_db_map)
    scenario = in_db_map.scenario_filter()
    if scenario is None:
        raise ScendesignerException()
    alternatives = in_db_map.scenario(name=scenario)["alternative_name_list"]
    analysis_parameters = read_analysis_parameters_for_scenario(
        config_db_map, in_db_map, alternatives
    )
    if isinstance(method, FullFactorialMethod):
        levels = read_individual_levels_for_scenario(
            config_db_map, alternatives, analysis_parameters.index
        )
        for i, individual_levels in enumerate(levels):
            if pd.isna(individual_levels):
                levels[i] = method.n_levels
        method = replace(method, n_levels=levels)
    sampling_data = generate_sampling(method, len(analysis_parameters))
    samples = apply_sampling(analysis_parameters, sampling_data)
    scenario_count = samples.shape[1]
    write_out_alternatives(out_db_map, scenario, scenario_count)
    write_samples(out_db_map, scenario, samples)
    write_scenarios(out_db_map, scenario, scenario_count, alternatives, scenario)
