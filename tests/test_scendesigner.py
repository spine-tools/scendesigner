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
import numpy as np
import pytest
import pandas as pd
import spinedb_api
import scendesigner


@pytest.fixture()
def in_db_map():
    with spinedb_api.DatabaseMapping("sqlite://", create=True) as db_map:
        yield db_map


@pytest.fixture()
def config_db_map():
    with spinedb_api.DatabaseMapping("sqlite://", create=True) as db_map:
        scendesigner.make_structure(db_map)
        yield db_map


@pytest.fixture()
def out_db_map():
    with spinedb_api.DatabaseMapping("sqlite://", create=True) as db_map:
        yield db_map


class TestMakeStructure:
    def test_creates_method_value_list(self, in_db_map):
        scendesigner.make_structure(in_db_map)
        value_lists = in_db_map.find_parameter_value_lists()
        assert len(value_lists) == 1
        value_list = value_lists[0]
        assert value_list["name"] == "SensitivityAnalysisMethod"
        assert value_list["parsed_value_list"] == ["Morris method"]

    def test_creates_entity_classes(self, in_db_map):
        scendesigner.make_structure(in_db_map)
        assert {
            entity_class["name"] for entity_class in in_db_map.find_entity_classes()
        } == {"SensitivityAnalysis", "SensitiveParameter"}

    def test_creates_sensitivity_analysis_parameters(self, in_db_map):
        scendesigner.make_structure(in_db_map)
        self.assert_structure(in_db_map)

    def test_rewriting_structure_succeeds(self, in_db_map):
        scendesigner.make_structure(in_db_map)
        scendesigner.make_structure(in_db_map)
        self.assert_structure(in_db_map)

    @staticmethod
    def assert_structure(db_map):
        assert (
            len(
                db_map.find_parameter_definitions(
                    entity_class_name="SensitivityAnalysis"
                )
            )
            == 5
        )
        version = db_map.parameter_definition(
            entity_class_name="SensitivityAnalysis", name="version"
        )
        assert version["parsed_value"] == 1
        assert version["parameter_type_list"] == ("float",)
        method = db_map.parameter_definition(
            entity_class_name="SensitivityAnalysis", name="method"
        )
        assert method["parameter_value_list_name"] == "SensitivityAnalysisMethod"
        random_seed = db_map.parameter_definition(
            entity_class_name="SensitivityAnalysis", name="random_seed"
        )
        assert random_seed["parsed_value"] == 1337
        assert random_seed["parameter_type_list"] == ("float",)
        trajectories = db_map.parameter_definition(
            entity_class_name="SensitivityAnalysis", name="trajectories"
        )
        assert trajectories["parsed_value"] == 10
        assert trajectories["parameter_type_list"] == ("float",)
        levels = db_map.parameter_definition(
            entity_class_name="SensitivityAnalysis", name="levels"
        )
        assert levels["parsed_value"] == 4
        assert levels["parameter_type_list"] == ("float",)


class TestAddEntityClassToAnalysis:
    def test_adds_class_with_correct_dimensions(self, config_db_map):
        config_db_map.add_entity_class(name="MyClass")
        scendesigner.add_entity_class_to_analysis(config_db_map, "MyClass")
        assert len(config_db_map.find_entity_classes()) == 4
        entity_class = config_db_map.entity_class(name="MyClass__SensitiveParameter")
        assert entity_class["dimension_name_list"] == ("MyClass", "SensitiveParameter")

    def test_adds_parameter_definition(self, config_db_map):
        config_db_map.add_entity_class(name="MyClass")
        scendesigner.add_entity_class_to_analysis(config_db_map, "MyClass")
        assert (
            len(
                config_db_map.find_parameter_definitions(
                    entity_class_name="MyClass__SensitiveParameter"
                )
            )
            == 1
        )
        variability_fraction = config_db_map.parameter_definition(
            entity_class_name="MyClass__SensitiveParameter", name="variability_fraction"
        )
        assert variability_fraction["parameter_type_list"] == ("float",)

    def test_rewriting_class_succeeds(self, config_db_map):
        config_db_map.add_entity_class(name="MyClass")
        scendesigner.add_entity_class_to_analysis(config_db_map, "MyClass")
        scendesigner.add_entity_class_to_analysis(config_db_map, "MyClass")
        assert (
            len(
                config_db_map.find_parameter_definitions(
                    entity_class_name="MyClass__SensitiveParameter"
                )
            )
            == 1
        )


class TestAddVariabilityFraction:
    def test_adds_entity_and_value(self, config_db_map):
        config_db_map.add_entity_class(name="MyClass")
        config_db_map.add_entity(entity_class_name="MyClass", name="a")
        scendesigner.add_parameter_to_analysis(config_db_map, "X")
        scendesigner.add_entity_class_to_analysis(config_db_map, "MyClass")
        scendesigner.add_entity_to_analysis(config_db_map, "MyClass", ("a",), "X")
        scendesigner.add_variability_fraction(
            config_db_map, "MyClass", ("a",), "X", "Base", 0.1
        )
        entities = config_db_map.find_entities(
            entity_class_name="MyClass__SensitiveParameter"
        )
        assert len(entities) == 1
        assert entities[0]["entity_byname"] == ("a", "X")
        parameters = config_db_map.find_parameter_values(
            entity_class_name="MyClass__SensitiveParameter", entity_byname=("a", "X")
        )
        assert len(parameters) == 1
        assert parameters[0]["parsed_value"] == 0.1
        assert parameters[0]["alternative_name"] == "Base"


class TestBuildAnalysis:
    def test_build_simple_analysis(self, config_db_map, in_db_map):
        scendesigner.make_structure(config_db_map)
        in_db_map.add_entity_class(name="fish")
        in_db_map.add_parameter_definition(entity_class_name="fish", name="speed")
        in_db_map.add_entity(entity_class_name="fish", name="Nemo")
        in_db_map.add_parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="Base",
            parsed_value=3.2,
        )
        in_db_map.add_scenario(name="MyScenario")
        in_db_map.add_scenario_alternative(
            scenario_name="MyScenario", alternative_name="Base", rank=0
        )
        in_db_map.commit_session("Add test data.")
        config_db_map.add_entity_class(name="fish")
        config_db_map.add_entity(entity_class_name="fish", name="Nemo")
        analysis_parameters = [
            scendesigner.ParameterAnalysisData("speed", scendesigner.Fractional(0.1))
        ]
        analysis_entities = [
            scendesigner.EntityAnalysisData(("Nemo",), analysis_parameters)
        ]
        analysis_data = [
            scendesigner.EntityClassAnalysisData("fish", analysis_entities)
        ]
        scendesigner.build_analysis(
            config_db_map, in_db_map, "MyScenario", analysis_data
        )
        assert in_db_map.is_committed()
        assert len(config_db_map.find_entity_classes()) == 4
        assert (
            len(
                config_db_map.find_entities(
                    entity_class_name="fish__SensitiveParameter"
                )
            )
            == 1
        )
        assert (
            len(
                config_db_map.find_parameter_values(
                    entity_class_name="fish__SensitiveParameter",
                    entity_byname=("Nemo", "speed"),
                )
            )
            == 1
        )
        value = config_db_map.parameter_value(
            entity_class_name="fish__SensitiveParameter",
            entity_byname=("Nemo", "speed"),
            parameter_definition_name="variability_fraction",
            alternative_name="Base",
        )
        assert value["parsed_value"] == 0.1


class TestReadMethod:
    def test_read_morris_method(self, config_db_map):
        method = scendesigner.scendesigner.read_method(config_db_map)
        assert (
            method.n_trajectories
            == config_db_map.parameter_definition(
                entity_class_name="SensitivityAnalysis", name="trajectories"
            )["parsed_value"]
        )
        assert isinstance(method.n_trajectories, int)
        assert (
            method.n_levels
            == config_db_map.parameter_definition(
                entity_class_name="SensitivityAnalysis", name="levels"
            )["parsed_value"]
        )
        assert isinstance(method.n_levels, int)
        assert (
            method.random_seed
            == config_db_map.parameter_definition(
                entity_class_name="SensitivityAnalysis", name="random_seed"
            )["parsed_value"]
        )
        assert isinstance(method.random_seed, int)


class TestApplySampling:
    def test_values(self):
        index = pd.MultiIndex.from_tuples(
            [("fish", ("Nemo",), "speed", "Base")],
            names=["entity_class", "entity", "parameter_definition", "alternative"],
        )
        parameters = pd.DataFrame(
            {"variability_fraction": [0.2], "value": [21.0]}, index=index
        )
        sampling = np.array([[0.5], [0.25], [0.75]])
        sampled = scendesigner.scendesigner.apply_sampling(parameters, sampling)
        expected = pd.DataFrame(
            {0: [21.0], 1: [21.0 - 2.1 / 2], 2: [21.0 + 2.1 / 2]}, index=index
        )
        pd.testing.assert_frame_equal(sampled, expected)


class TestReadAnalysisParametersForScenario:
    def test_reads_single_parameter(self, config_db_map, in_db_map):
        in_db_map.add_entity_class(name="fish")
        in_db_map.add_parameter_definition(entity_class_name="fish", name="speed")
        in_db_map.add_entity(entity_class_name="fish", name="Nemo")
        in_db_map.add_parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="Base",
            parsed_value=2.3,
        )
        config_db_map.add_entity_class(name="fish")
        config_db_map.add_entity(entity_class_name="fish", name="Nemo")
        scendesigner.add_entity_class_to_analysis(config_db_map, "fish")
        scendesigner.add_parameter_to_analysis(config_db_map, "speed")
        scendesigner.add_entity_to_analysis(config_db_map, "fish", ("Nemo",), "speed")
        scendesigner.add_variability_fraction(
            config_db_map, "fish", ("Nemo",), "speed", "Base", 0.1
        )
        parameters = scendesigner.scendesigner.read_analysis_parameters_for_scenario(
            config_db_map, in_db_map, ["Base"]
        )
        index = pd.MultiIndex.from_tuples(
            [("fish", ("Nemo",), "speed")],
            names=["entity_class", "entity", "parameter_definition"],
        )
        expected = pd.DataFrame(
            {"variability_fraction": [0.1], "value": [2.3]}, index=index
        )
        pd.testing.assert_frame_equal(parameters, expected)

    def test_value_in_highest_ranking_alternative_wins(self, config_db_map, in_db_map):
        in_db_map.add_alternative(name="High")
        in_db_map.add_entity_class(name="fish")
        in_db_map.add_parameter_definition(entity_class_name="fish", name="speed")
        in_db_map.add_entity(entity_class_name="fish", name="Nemo")
        in_db_map.add_parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="Base",
            parsed_value=2.3,
        )
        in_db_map.add_parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="High",
            parsed_value=3.2,
        )
        config_db_map.add_entity_class(name="fish")
        config_db_map.add_entity(entity_class_name="fish", name="Nemo")
        config_db_map.add_alternative(name="High")
        scendesigner.add_entity_class_to_analysis(config_db_map, "fish")
        scendesigner.add_parameter_to_analysis(config_db_map, "speed")
        scendesigner.add_entity_to_analysis(config_db_map, "fish", ("Nemo",), "speed")
        scendesigner.add_variability_fraction(
            config_db_map, "fish", ("Nemo",), "speed", "Base", 0.1
        )
        scendesigner.add_variability_fraction(
            config_db_map, "fish", ("Nemo",), "speed", "High", 0.2
        )
        parameters = scendesigner.scendesigner.read_analysis_parameters_for_scenario(
            config_db_map, in_db_map, ["Base", "High"]
        )
        index = pd.MultiIndex.from_tuples(
            [("fish", ("Nemo",), "speed")],
            names=["entity_class", "entity", "parameter_definition"],
        )
        expected = pd.DataFrame(
            {"variability_fraction": [0.2], "value": [3.2]}, index=index
        )
        pd.testing.assert_frame_equal(parameters, expected)


class TestWriteVariedValues:
    def test_write_single_parameter(self, out_db_map):
        out_db_map.add_entity_class(name="fish")
        out_db_map.add_entity(entity_class_name="fish", entity_byname=("Nemo",))
        out_db_map.add_parameter_definition(entity_class_name="fish", name="speed")
        index = pd.MultiIndex.from_tuples(
            [("fish", ("Nemo",), "speed")],
            names=["entity_class", "entity", "parameter_definition"],
        )
        samples = pd.DataFrame(
            {0: [21.0], 1: [21.0 - 2.1 / 2], 2: [21.0 + 2.1 / 2]}, index=index
        )
        scendesigner.scendesigner.write_out_alternatives(out_db_map, "sample", 3)
        scendesigner.scendesigner.write_samples(out_db_map, "sample", samples)
        assert len(out_db_map.find_parameter_values()) == 3
        value = out_db_map.parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="sample_001",
        )
        assert value["parsed_value"] == 21.0
        value = out_db_map.parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="sample_002",
        )
        assert value["parsed_value"] == 21.0 - 2.1 / 2
        value = out_db_map.parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="sample_003",
        )
        assert value["parsed_value"] == 21.0 + 2.1 / 2


class TestWriteScenarios:
    def test_scenario_has_required_alternatives(self, out_db_map):
        scendesigner.scendesigner.write_out_alternatives(out_db_map, "alt", 1)
        scendesigner.scendesigner.write_scenarios(
            out_db_map, "MyScenario", 1, ["Base"], "alt"
        )
        assert len(out_db_map.find_scenarios()) == 1
        assert len(out_db_map.find_alternatives()) == 2
        scenario = out_db_map.scenario(name="MyScenario_001")
        assert scenario["description"] == "For sensitivity analysis."
        assert scenario["alternative_name_list"] == ["Base", "alt_001"]


class TestMakeScenarios:
    def test_data_gets_written_to_output_database(
        self, config_db_map, in_db_map, out_db_map
    ):
        scendesigner.make_structure(config_db_map)
        in_db_map.add_entity_class(name="fish")
        in_db_map.add_parameter_definition(entity_class_name="fish", name="speed")
        in_db_map.add_entity(entity_class_name="fish", name="Nemo")
        in_db_map.add_parameter_value(
            entity_class_name="fish",
            entity_byname=("Nemo",),
            parameter_definition_name="speed",
            alternative_name="Base",
            parsed_value=3.2,
        )
        in_db_map.add_scenario(name="MyScenario")
        in_db_map.add_scenario_alternative(
            scenario_name="MyScenario", alternative_name="Base", rank=0
        )
        in_db_map.commit_session("Add test data.")
        spinedb_api.filters.tools.apply_filter_stack(
            in_db_map,
            [spinedb_api.filters.scenario_filter.scenario_filter_config("MyScenario")],
        )
        config_db_map.add_entity_class(name="fish")
        config_db_map.add_entity(entity_class_name="fish", name="Nemo")
        out_db_map.add_entity_class(name="fish")
        out_db_map.add_parameter_definition(entity_class_name="fish", name="speed")
        out_db_map.add_entity(entity_class_name="fish", name="Nemo")
        analysis_data = [
            scendesigner.EntityClassAnalysisData(
                "fish",
                [
                    scendesigner.EntityAnalysisData(
                        ("Nemo",),
                        [
                            scendesigner.ParameterAnalysisData(
                                "speed", scendesigner.Fractional(0.1)
                            )
                        ],
                    )
                ],
            )
        ]
        scendesigner.build_analysis(
            config_db_map, in_db_map, "MyScenario", analysis_data
        )
        config_db_map.commit_session("Add config data.")
        scendesigner.make_scenarios(config_db_map, in_db_map, out_db_map)
        assert in_db_map.is_committed()
        assert config_db_map.is_committed()
        assert not out_db_map.is_committed()
        method = scendesigner.scendesigner.read_method(config_db_map)
        analysed_parameter_count = 1
        expected_scenario_count = method.n_trajectories * (analysed_parameter_count + 1)
        assert len(out_db_map.find_alternatives()) == 1 + expected_scenario_count
        assert len(out_db_map.find_scenarios()) == expected_scenario_count
        for i in range(expected_scenario_count):
            assert out_db_map.scenario(name=f"MyScenario_{i + 1:03}")[
                "alternative_name_list"
            ] == ["Base", f"MyScenario_{i + 1:03}"]
        assert len(out_db_map.find_entity_classes()) == 1
        assert len(out_db_map.find_parameter_definitions()) == 1
        assert len(out_db_map.find_entities(entity_class_name="fish")) == 1
        assert (
            len(
                out_db_map.find_parameter_values(
                    entity_class_name="fish", parameter_definition_name="speed"
                )
            )
            == expected_scenario_count
        )
