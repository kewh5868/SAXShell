from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QMessageBox,
)

import saxshell.saxs.project_manager.project as project_module
from saxshell.saxs._model_templates import (
    load_template_module,
    load_template_spec,
)
from saxshell.saxs.dream import (
    DreamParameterEntry,
    DreamRunSettings,
    SAXSDreamResultsLoader,
    SAXSDreamWorkflow,
)
from saxshell.saxs.prefit import SAXSPrefitWorkflow
from saxshell.saxs.project_manager import (
    ClusterImportResult,
    ExperimentalDataSummary,
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
    load_experimental_data_file,
)
from saxshell.saxs.ui.distribution_window import DistributionSetupWindow
from saxshell.saxs.ui.experimental_data_loader import (
    ExperimentalDataHeaderDialog,
)
from saxshell.saxs.ui.main_window import RuntimeBundleOpener, SAXSMainWindow
from saxshell.saxs.ui.prior_histogram_window import PriorHistogramWindow
from saxshell.saxs.ui.project_setup_tab import ProjectSetupTab
from saxshell.version import __version__


def _write_component_file(path, q_values, intensities):
    data = np.column_stack(
        [
            q_values,
            intensities,
            np.zeros_like(q_values),
            np.zeros_like(q_values),
        ]
    )
    np.savetxt(
        path,
        data,
        header="# Number of files: 1\n# Columns: q, S(q)_avg, S(q)_std, S(q)_se",
        comments="",
    )


def _build_minimal_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)

    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)
    template_name = "template_pd_likelihood_monosq_decoupled"
    template_module = load_template_module(template_name)
    experimental = template_module.lmfit_model_profile(
        q_values,
        np.zeros_like(q_values),
        [component],
        w0=0.6,
        solv_w=0.0,
        offset=0.05,
        eff_r=9.0,
        vol_frac=0.0,
        scale=5e-4,
    )

    experimental_path = paths.experimental_data_dir / "exp_demo.txt"
    np.savetxt(experimental_path, np.column_stack([q_values, experimental]))
    _write_component_file(
        paths.scattering_components_dir / "A_no_motif.txt",
        q_values,
        component,
    )
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 1,
                "structures": {
                    "A": {
                        "no_motif": {
                            "count": 1,
                            "weight": 0.6,
                            "representative": "frame_0001.xyz",
                            "profile_file": "A_no_motif.txt",
                        }
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.project_dir / "md_saxs_map.json").write_text(
        json.dumps(
            {"saxs_map": {"A": {"no_motif": "A_no_motif.txt"}}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    settings.experimental_data_path = str(experimental_path)
    settings.copied_experimental_data_file = str(experimental_path)
    settings.selected_model_template = template_name
    manager.save_project(settings)
    return project_dir, paths


def _write_minimal_dream_results(project_dir):
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map()
    bundle = workflow.create_runtime_bundle(entries=entries)

    active_values = np.asarray(
        [entry.value for entry in entries if entry.vary],
        dtype=float,
    )
    sampled_params = []
    log_ps = []
    for chain_index in range(2):
        chain_samples = []
        chain_logps = []
        for step_index in range(4):
            adjustment = (chain_index + 1) * (step_index + 1) * 0.01
            chain_samples.append(active_values + adjustment)
            chain_logps.append(-5.0 + chain_index + step_index * 0.5)
        sampled_params.append(chain_samples)
        log_ps.append(chain_logps)
    np.save(
        bundle.run_dir / "dream_sampled_params.npy", np.asarray(sampled_params)
    )
    np.save(bundle.run_dir / "dream_log_ps.npy", np.asarray(log_ps))
    return bundle


def _write_weight_order_dream_results(tmp_path):
    run_dir = tmp_path / "dream_order_test"
    run_dir.mkdir(parents=True)
    metadata = {
        "settings": {"burnin_percent": 0},
        "template_name": "template_pd_likelihood_monosq_decoupled",
        "parameter_map": [
            {
                "structure": "PbI2O",
                "motif": "m3",
                "param_type": "Both",
                "param": "w2",
                "value": 0.25,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {"loc": 0.0, "scale": 0.25, "s": 0.1},
            },
            {
                "structure": "I2",
                "motif": "m1",
                "param_type": "Both",
                "param": "w0",
                "value": 0.35,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {"loc": 0.0, "scale": 0.35, "s": 0.1},
            },
            {
                "structure": "Pb2",
                "motif": "m2",
                "param_type": "Both",
                "param": "w1",
                "value": 0.4,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {"loc": 0.0, "scale": 0.4, "s": 0.1},
            },
            {
                "structure": "",
                "motif": "",
                "param_type": "SAXS",
                "param": "scale",
                "value": 1.0,
                "vary": False,
                "distribution": "norm",
                "dist_params": {"loc": 1.0, "scale": 0.1},
            },
        ],
        "active_parameter_entries": [
            {
                "structure": "PbI2O",
                "motif": "m3",
                "param_type": "Both",
                "param": "w2",
                "value": 0.25,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {"loc": 0.0, "scale": 0.25, "s": 0.1},
            },
            {
                "structure": "I2",
                "motif": "m1",
                "param_type": "Both",
                "param": "w0",
                "value": 0.35,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {"loc": 0.0, "scale": 0.35, "s": 0.1},
            },
            {
                "structure": "Pb2",
                "motif": "m2",
                "param_type": "Both",
                "param": "w1",
                "value": 0.4,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {"loc": 0.0, "scale": 0.4, "s": 0.1},
            },
        ],
        "active_parameter_indices": [0, 1, 2],
        "full_parameter_names": ["w2", "w0", "w1", "scale"],
        "fixed_parameter_values": [0.25, 0.35, 0.4, 1.0],
        "q_values": [0.1, 0.2],
        "experimental_intensities": [1.0, 0.8],
        "theoretical_intensities": [[1.0, 0.9]],
        "solvent_intensities": [0.0, 0.0],
    }
    (run_dir / "dream_runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(
        run_dir / "dream_sampled_params.npy",
        np.asarray(
            [
                [
                    [0.25, 0.35, 0.40],
                    [0.26, 0.34, 0.41],
                ]
            ],
            dtype=float,
        ),
    )
    np.save(
        run_dir / "dream_log_ps.npy",
        np.asarray([[1.0, 2.0]], dtype=float),
    )
    return run_dir


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_saxs_main_window_loads_project_prefit_and_dream_tabs(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    assert window.project_setup_tab.forward_model_group.isEnabled()
    assert window.project_setup_tab.model_group.isEnabled()
    assert window.project_setup_tab.template_combo.count() >= 1
    assert (
        window.project_setup_tab.selected_template_name()
        == "template_pd_likelihood_monosq_decoupled"
    )
    assert window.project_setup_tab.template_combo.currentText() == (
        "MonoSQ Decoupled"
    )
    assert (
        "Structure Factor:"
        in window.project_setup_tab.template_combo.toolTip()
    )
    assert (
        "keith.white@colorado.edu"
        in window.project_setup_tab.template_help_button.toolTip()
    )
    assert window.project_setup_tab.project_name_edit.text() == "saxs_project"
    assert window.project_setup_tab.component_log_x_checkbox.isChecked()
    assert window.project_setup_tab.component_log_y_checkbox.isChecked()
    assert window.project_setup_tab.project_dir_edit.text() == str(
        project_dir.parent
    )
    assert window.project_setup_tab.open_project_dir_edit.text().endswith(
        "saxs_project"
    )
    assert window.prefit_tab.parameter_table.rowCount() == 6
    assert window.prefit_tab.template_combo.count() >= 1
    assert window.prefit_tab.selected_template_name() == (
        "template_pd_likelihood_monosq_decoupled"
    )
    assert window.prefit_tab.template_combo.currentText() == (
        "MonoSQ Decoupled"
    )
    assert "Form Factor:" in window.prefit_tab.template_combo.toolTip()
    assert (
        "keith.white@colorado.edu"
        in window.prefit_tab.template_help_button.toolTip()
    )
    assert "Refine scale before the other model parameters" in (
        window.prefit_tab.prefit_help_button.toolTip()
    )
    assert window.prefit_tab.log_x_checkbox.isChecked()
    assert window.prefit_tab.log_y_checkbox.isChecked()
    assert window.prefit_tab.plot_toolbar is not None
    assert not window.prefit_tab.autosave_checkbox.isChecked()
    assert window.prefit_tab.recommended_scale_button is not None
    assert window.prefit_tab.set_best_button is not None
    assert window.prefit_tab.reset_best_button is not None
    assert window.prefit_tab.restore_state_button is not None
    assert not window.prefit_tab.restore_state_button.isEnabled()
    assert "Best Prefit preset" in window.prefit_tab.set_best_button.toolTip()
    assert "template-default prefit preset" in (
        window.prefit_tab.reset_button.toolTip()
    )
    assert "timestamped prefit snapshot folders" in (
        window.prefit_tab.saved_state_combo.toolTip()
    )
    assert "Prefit summary:" in window.prefit_tab.summary_box.toPlainText()
    assert (
        "Template: template_pd_likelihood_monosq_decoupled"
        in window.prefit_tab.summary_box.toPlainText()
    )
    assert "Prefit Console" in window.prefit_tab.summary_box.toPlainText()
    assert window.dream_tab.chains_spin.value() == 4
    assert (
        window.dream_tab.settings_preset_combo.currentText()
        == window.dream_tab.ACTIVE_SETTINGS_LABEL
    )
    assert window.dream_tab.model_name_edit is not None
    assert window.dream_tab.nseedchains_spin.value() == 40
    assert window.dream_tab.output_box is window.dream_tab.log_box
    assert window.dream_tab.output_box is window.dream_tab.summary_box
    assert window.dream_tab.model_toolbar is not None
    assert window.dream_tab.violin_toolbar is not None
    assert window.dream_tab.settings_preset_combo.toolTip()
    assert window.dream_tab.model_name_edit.toolTip()
    assert window.dream_tab.chains_spin.toolTip()
    assert window.dream_tab.run_button.toolTip()
    assert "DREAM Summary" in window.dream_tab.output_box.toPlainText()
    assert "DREAM Console" in window.dream_tab.output_box.toPlainText()


def test_main_window_menus_expose_project_tools_and_help(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    assert window.file_menu.title() == "File"
    assert window.create_project_action.text() == "Create Project"
    assert window.open_project_action.text() == "Open Existing Project..."
    assert window.open_recent_menu.title() == "Open Recent Project"
    assert window.save_project_action.text() == "Save Project"
    assert window.save_project_action.shortcuts()
    assert window.save_project_as_action.text() == "Save Project As..."

    assert window.tools_menu.title() == "Tools"
    assert window.mdtrajectory_action.text() == "Open mdtrajectory"
    assert window.cluster_action.text() == "Open Cluster Extraction"
    assert window.xyz2pdb_action.text() == "Open xyz2pdb Conversion"
    assert window.bondanalysis_action.text() == "Open Bond Analysis"

    assert window.settings_menu.title() == "Settings"
    assert (
        window.dream_output_settings_action.text()
        == "DREAM Output Settings..."
    )

    assert window.help_menu.title() == "Help"
    assert window.version_info_action.text() == "Version Information"
    assert window.github_action.text() == "Open GitHub Repository"
    assert window.contact_action.text() == "Contact Developer"

    version_text = window._version_information_text()
    assert __version__
    assert "Package version:" in version_text
    assert "GitHub repository:" in version_text
    assert "Developer contact:" in version_text
    assert "keith.white@colorado.edu" in version_text


def test_contact_action_opens_developer_contact_window(qapp, monkeypatch):
    del qapp
    window = SAXSMainWindow()
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.QMessageBox.information",
        lambda parent, title, message: captured.update(
            {
                "title": title,
                "message": message,
            }
        ),
    )

    window._show_contact_information()

    assert captured["title"] == "Developer Contact"
    assert "contact the developer" in captured["message"].lower()
    assert "keith.white@colorado.edu" in captured["message"]


def test_bondanalysis_tool_uses_active_project_clusters_dir(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    launched: dict[str, object] = {}

    class FakeBondAnalysisWindow:
        def __init__(self, initial_clusters_dir=None):
            launched["clusters_dir"] = initial_clusters_dir
            launched["instance"] = self

        def show(self):
            launched["shown"] = True

        def raise_(self):
            launched["raised"] = True

    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.main_window.BondAnalysisMainWindow",
        FakeBondAnalysisWindow,
    )

    window._open_bondanalysis_tool()

    assert launched["clusters_dir"] == window.current_settings.clusters_dir
    assert launched["shown"] is True
    assert launched["raised"] is True
    assert launched["instance"] in window._child_tool_windows


def test_save_project_as_copies_project_and_rewrites_internal_paths(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    destination_parent = tmp_path / "copied_projects"
    destination_parent.mkdir()

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(destination_parent),
    )
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args, **kwargs: ("renamed_project", True),
    )

    window.save_project_as()

    copied_dir = destination_parent / "renamed_project"
    copied_settings = SAXSProjectManager().load_project(copied_dir)

    assert copied_dir.is_dir()
    assert copied_settings.project_name == "renamed_project"
    assert copied_settings.project_dir == str(copied_dir.resolve())
    assert copied_settings.copied_experimental_data_file == str(
        (copied_dir / "experimental_data" / "exp_demo.txt").resolve()
    )
    assert copied_settings.experimental_data_path == str(
        (copied_dir / "experimental_data" / "exp_demo.txt").resolve()
    )
    assert window.current_settings is not None
    assert window.current_settings.project_dir == str(copied_dir.resolve())


def test_prefit_layout_uses_left_parameter_panel_and_combined_output(qapp):
    del qapp
    window = SAXSMainWindow()

    assert window.prefit_tab.parameter_table.parentWidget() is not None
    assert window.prefit_tab._scroll_area.widget() is not None
    assert window.prefit_tab._scroll_area.widgetResizable()
    assert window.prefit_tab._main_splitter.count() == 2
    assert (
        window.prefit_tab._main_splitter.widget(0)
        is window.prefit_tab._pane_splitter
    )
    assert (
        window.prefit_tab._main_splitter.widget(1)
        is window.prefit_tab._output_group
    )
    assert window.prefit_tab._pane_splitter.count() == 2
    assert (
        window.prefit_tab._pane_splitter.widget(0)
        is window.prefit_tab._left_panel
    )
    assert (
        window.prefit_tab._pane_splitter.widget(1)
        is window.prefit_tab._plot_group
    )
    assert window.prefit_tab.output_box is window.prefit_tab.log_box
    assert window.prefit_tab.output_box is window.prefit_tab.summary_box
    assert window.prefit_tab.update_button.parentWidget() is not None
    assert window.prefit_tab.run_button.parentWidget() is not None
    assert window.prefit_tab.save_button.parentWidget() is not None
    assert window.prefit_tab.reset_button.parentWidget() is not None
    assert window.prefit_tab.set_best_button.parentWidget() is not None
    assert window.prefit_tab.reset_best_button.parentWidget() is not None
    assert window.prefit_tab.restore_state_button.parentWidget() is not None


def test_dream_layout_uses_combined_output_and_two_plot_panels(qapp):
    del qapp
    window = SAXSMainWindow()

    assert window.dream_tab._outer_scroll_area.widget() is not None
    assert window.dream_tab._outer_scroll_area.widgetResizable()
    assert window.dream_tab.output_box is window.dream_tab.log_box
    assert window.dream_tab.output_box is window.dream_tab.summary_box
    assert window.dream_tab._main_splitter is window.dream_tab._top_splitter
    assert window.dream_tab._main_splitter.count() == 2
    assert window.dream_tab.verbose_checkbox.isChecked()
    assert window.dream_tab.verbose_interval_spin.value() == pytest.approx(1.0)
    assert window.dream_tab.verbose_interval_spin.isEnabled()
    assert window.dream_tab.selected_search_filter_preset() == "medium"
    assert window.dream_tab.chains_spin.value() == 4
    assert window.dream_tab.iterations_spin.value() == 10000
    assert window.dream_tab.burnin_spin.value() == 20
    assert window.dream_tab.history_thin_spin.value() == 10
    assert window.dream_tab.nseedchains_spin.value() == 40
    assert window.dream_tab.lambda_spin.value() == pytest.approx(0.05)
    assert window.dream_tab.zeta_spin.value() == pytest.approx(1e-12)
    assert window.dream_tab.snooker_spin.value() == pytest.approx(0.1)
    assert window.dream_tab.p_gamma_unity_spin.value() == pytest.approx(0.2)
    assert (
        window.dream_tab._main_splitter.widget(0)
        is window.dream_tab._settings_scroll_area
    )
    assert (
        window.dream_tab._main_splitter.widget(1)
        is window.dream_tab._plot_scroll_area
    )
    assert (
        window.dream_tab._plot_scroll_area.widget()
        is window.dream_tab._plot_panel
    )
    assert window.dream_tab._plot_splitter.count() == 2
    assert window.dream_tab._output_group.parentWidget() is not None
    assert window.dream_tab.model_canvas is not None
    assert window.dream_tab.violin_canvas is not None
    assert (
        window.dream_tab._plot_splitter.widget(0)
        is window.dream_tab.model_canvas.parentWidget()
    )
    assert (
        window.dream_tab._plot_splitter.widget(1)
        is window.dream_tab.violin_canvas.parentWidget()
    )
    assert (
        window.dream_tab.posterior_filter_combo.currentData()
        == "all_post_burnin"
    )
    assert not window.dream_tab.posterior_top_percent_spin.isEnabled()
    assert not window.dream_tab.posterior_top_n_spin.isEnabled()
    assert (
        window.dream_tab.violin_sample_source_combo.currentData()
        == "filtered_posterior"
    )
    assert window.dream_tab.weight_order_combo.currentData() == "weight_index"
    assert (
        window.dream_tab.violin_value_scale_combo.currentData()
        == "parameter_value"
    )
    assert window.dream_tab.violin_palette_combo.currentData() == "Blues"
    assert window.dream_tab.selected_violin_point_color() == to_hex(
        "tab:red",
        keep_alpha=False,
    )
    assert not window.dream_tab.violin_custom_color_button.isEnabled()
    assert window.dream_tab.violin_point_color_button.isEnabled()
    assert (
        "highest-log-posterior samples"
        in window.dream_tab.posterior_top_percent_spin.toolTip()
    )
    assert window.dream_tab.parameter_map_table.rowCount() == 0
    assert (
        window.dream_tab.parameter_map_table.editTriggers()
        == window.dream_tab.parameter_map_table.EditTrigger.NoEditTriggers
    )
    assert window.dream_tab.edit_button.parentWidget() is not None
    assert window.dream_tab.save_settings_button.parentWidget() is not None
    assert window.dream_tab.write_button.parentWidget() is not None
    assert window.dream_tab.preview_button.parentWidget() is not None
    assert window.dream_tab.run_button.parentWidget() is not None
    assert window.dream_tab.load_button.parentWidget() is not None
    assert window.dream_tab.report_button.parentWidget() is not None
    assert window.dream_tab.save_model_button.parentWidget() is not None
    assert window.dream_tab.save_violin_button.parentWidget() is not None
    assert window.dream_tab.setup_actions_group.title() == "DREAM Setup"
    assert window.dream_tab.analysis_actions_group.title() == (
        "DREAM Analysis"
    )


def test_dream_zeta_spin_uses_scientific_notation(qapp):
    del qapp
    window = SAXSMainWindow()

    window.dream_tab.set_settings(DreamRunSettings(zeta=1e-12))

    assert "e" in window.dream_tab.zeta_spin.text().lower()
    assert window.dream_tab.zeta_spin.text().lower().endswith("e-12")
    assert window.dream_tab.settings_payload().zeta == pytest.approx(1e-12)


def test_apply_dream_output_settings_updates_verbose_controls(qapp):
    del qapp
    window = SAXSMainWindow()

    window._apply_dream_output_settings(verbose=False, interval_seconds=2.5)

    assert not window.dream_tab.verbose_checkbox.isChecked()
    assert window.dream_tab.verbose_interval_spin.value() == pytest.approx(2.5)
    assert not window.dream_tab.verbose_interval_spin.isEnabled()
    assert "Updated DREAM output settings." in (
        window.dream_tab.output_box.toPlainText()
    )


def test_dream_search_filter_presets_update_and_fall_back_to_custom(qapp):
    del qapp
    window = SAXSMainWindow()

    aggressive_index = window.dream_tab.search_filter_preset_combo.findData(
        "more_aggressive"
    )
    window.dream_tab.search_filter_preset_combo.setCurrentIndex(
        aggressive_index
    )

    assert (
        window.dream_tab.selected_search_filter_preset() == "more_aggressive"
    )
    assert window.dream_tab.chains_spin.value() == 8
    assert window.dream_tab.iterations_spin.value() == 20000
    assert window.dream_tab.burnin_spin.value() == 25
    assert window.dream_tab.posterior_filter_combo.currentData() == (
        "top_percent_logp"
    )
    assert (
        window.dream_tab.posterior_top_percent_spin.value()
        == pytest.approx(5.0)
    )

    window.dream_tab.iterations_spin.setValue(21000)

    assert window.dream_tab.selected_search_filter_preset() == "custom"


def test_main_window_ui_scale_updates_fonts_and_minimum_sizes(qapp):
    del qapp
    window = SAXSMainWindow()

    base_font_size = window.font().pointSizeF()
    base_output_height = window.prefit_tab.output_box.minimumHeight()
    base_handle_width = window.dream_tab._top_splitter.handleWidth()

    window._set_ui_scale(1.2)

    assert window._ui_scale == 1.2
    assert window.font().pointSizeF() > base_font_size
    assert window.prefit_tab.output_box.minimumHeight() > base_output_height
    assert window.dream_tab._top_splitter.handleWidth() > base_handle_width

    window._reset_ui_scale()

    assert window._ui_scale == 1.0
    assert window.font().pointSizeF() == pytest.approx(base_font_size)
    assert window.prefit_tab.output_box.minimumHeight() == base_output_height
    assert window.dream_tab._top_splitter.handleWidth() == base_handle_width


def test_dream_blink_highlight_does_not_override_button_stylesheet(qapp):
    del qapp
    window = SAXSMainWindow()

    window.dream_tab.blink_edit_priors_button()

    assert window.dream_tab.edit_button.styleSheet() == ""
    assert window.dream_tab.edit_button.graphicsEffect() is not None


def test_dream_blink_uses_per_button_effects(qapp):
    del qapp
    window = SAXSMainWindow()

    window.dream_tab.blink_edit_priors_button()
    first_effect = window.dream_tab.edit_button.graphicsEffect()
    window.dream_tab.blink_write_bundle_button()

    assert window.dream_tab.edit_button.graphicsEffect() is None
    assert window.dream_tab.write_button.graphicsEffect() is not None
    assert window.dream_tab.write_button.graphicsEffect() is not first_effect


def test_dream_prior_map_table_updates_after_saving_entries(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    entries = [
        DreamParameterEntry(
            structure="PbI2",
            motif="motif_A",
            param_type="SAXS",
            param="w0",
            value=0.6,
            vary=True,
            distribution="lognorm",
            dist_params={"loc": 0.0, "scale": 0.6, "s": 0.1},
        )
    ]

    window._save_distribution_entries(entries)

    table = window.dream_tab.parameter_map_table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "PbI2"
    assert table.item(0, 3).text() == "w0"
    assert table.item(0, 5).text() == "Yes"


def test_distribution_window_prompts_before_quitting_without_first_save(
    qapp, monkeypatch
):
    del qapp
    window = DistributionSetupWindow(
        [
            DreamParameterEntry(
                structure="PbI2",
                motif="motif_A",
                param_type="SAXS",
                param="w0",
                value=0.6,
                vary=True,
                distribution="lognorm",
                dist_params={"loc": 0.0, "scale": 0.6, "s": 0.1},
            )
        ]
    )
    window.load_entries(
        window.current_entries(), has_existing_parameter_map=False
    )
    window.show()

    monkeypatch.setattr(
        "saxshell.saxs.ui.distribution_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )

    window.close()
    QApplication.processEvents()

    assert window.isVisible()

    monkeypatch.setattr(
        "saxshell.saxs.ui.distribution_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    window.close()
    QApplication.processEvents()

    assert not window.isVisible()


def test_distribution_window_switches_lognorm_params_for_norm_and_uniform(
    qapp,
):
    del qapp
    window = DistributionSetupWindow(
        [
            DreamParameterEntry(
                structure="PbI2",
                motif="motif_A",
                param_type="SAXS",
                param="w0",
                value=0.6,
                vary=True,
                distribution="lognorm",
                dist_params={"loc": 0.0, "scale": 0.6, "s": 0.1},
            )
        ]
    )
    combo = window.table.cellWidget(0, 6)

    combo.setCurrentText("norm")
    QApplication.processEvents()

    norm_params = json.loads(window.table.item(0, 7).text())
    assert "s" not in norm_params
    assert set(norm_params) == {"loc", "scale"}
    assert window.figure.axes[0].get_title() == "w0: norm"

    combo.setCurrentText("uniform")
    QApplication.processEvents()

    uniform_params = json.loads(window.table.item(0, 7).text())
    assert "s" not in uniform_params
    assert set(uniform_params) == {"loc", "scale"}
    assert window.figure.axes[0].get_title() == "w0: uniform"


def test_distribution_window_can_toggle_all_vary_flags(qapp):
    del qapp
    window = DistributionSetupWindow(
        [
            DreamParameterEntry(
                structure="PbI2",
                motif="motif_A",
                param_type="SAXS",
                param="w0",
                value=0.6,
                vary=True,
                distribution="lognorm",
                dist_params={"loc": 0.0, "scale": 0.6, "s": 0.1},
            ),
            DreamParameterEntry(
                structure="PbI2",
                motif="motif_B",
                param_type="SAXS",
                param="scale",
                value=1.0,
                vary=False,
                distribution="norm",
                dist_params={"loc": 1.0, "scale": 0.2},
            ),
        ]
    )

    window.set_all_vary_off_button.click()
    assert all(not entry.vary for entry in window.current_entries())

    window.set_all_vary_on_button.click()
    assert all(entry.vary for entry in window.current_entries())


def test_distribution_window_previews_all_weight_priors_in_shared_plot(qapp):
    del qapp
    window = DistributionSetupWindow(
        [
            DreamParameterEntry(
                structure="PbI2",
                motif="motif_A",
                param_type="Both",
                param="w0",
                value=0.6,
                vary=True,
                distribution="lognorm",
                dist_params={"loc": 0.0, "scale": 0.6, "s": 0.1},
            ),
            DreamParameterEntry(
                structure="PbI2",
                motif="motif_B",
                param_type="Both",
                param="w1",
                value=0.4,
                vary=True,
                distribution="norm",
                dist_params={"loc": 0.4, "scale": 0.05},
            ),
            DreamParameterEntry(
                structure="",
                motif="",
                param_type="SAXS",
                param="scale",
                value=1.0,
                vary=True,
                distribution="norm",
                dist_params={"loc": 1.0, "scale": 0.2},
            ),
        ]
    )

    window.preview_weight_priors_button.click()

    assert window._weight_preview_window is not None
    assert window._weight_preview_window.isVisible()
    axis = window._weight_preview_window.figure.axes[0]
    assert axis.get_title() == "Weight prior distributions"
    assert axis.get_xlabel() == "Value"
    assert axis.get_ylabel() == "Density"
    plotted_labels = [line.get_label() for line in axis.get_lines()]
    assert plotted_labels == ["w0 (PbI2)", "w1 (PbI2)"]


def test_template_dropdowns_use_display_names_and_tooltips(qapp):
    del qapp
    tab = ProjectSetupTab()
    basic_spec = load_template_spec("template_likelihood_monosq")
    decoupled_spec = load_template_spec(
        "template_pd_likelihood_monosq_decoupled"
    )

    tab.set_available_templates([basic_spec, decoupled_spec], basic_spec.name)

    assert tab.template_combo.itemText(0) == "MonoSQ Basic"
    assert (
        tab.template_combo.itemData(0, Qt.ItemDataRole.ToolTipRole)
        == basic_spec.description
    )
    assert tab.selected_template_name() == basic_spec.name


def test_project_setup_empty_preview_message_is_wrapped(qapp):
    del qapp
    tab = ProjectSetupTab()
    tab.draw_component_plot(None)

    preview_axis = tab.component_figure.axes[0]
    text_labels = [text.get_text() for text in preview_axis.texts]

    assert any(
        "Select experimental data and build SAXS" in label
        and "averaged cluster profiles." in label
        and "\n" in label
        for label in text_labels
    )


def test_run_dream_requires_parameter_map_saved_in_session(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    workflow.create_default_parameter_map()
    window = SAXSMainWindow(initial_project_dir=project_dir)

    blinked = {"value": False}
    monkeypatch.setattr(
        window.dream_tab,
        "blink_edit_priors_button",
        lambda: blinked.update({"value": True}),
    )
    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.SAXSDreamWorkflow.run_bundle",
        lambda self, bundle: (_ for _ in ()).throw(
            AssertionError("run_bundle should not be called")
        ),
    )

    window.run_dream_bundle()

    assert blinked["value"]
    assert "Review the priors in Edit Priors" in (
        window.dream_tab.output_box.toPlainText()
    )


def test_run_dream_requires_written_runtime_bundle(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    window._save_distribution_entries(entries)

    blinked = {"value": False}
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window.dream_tab,
        "blink_write_bundle_button",
        lambda: blinked.update({"value": True}),
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.SAXSDreamWorkflow.run_bundle",
        lambda self, bundle: (_ for _ in ()).throw(
            AssertionError("run_bundle should not be called")
        ),
    )

    window.run_dream_bundle()

    assert blinked["value"]
    assert errors == [
        (
            "Runtime Bundle not generated",
            "Runtime Bundle not generated. Click Write Runtime Bundle before running DREAM.",
        )
    ]
    assert "Runtime Bundle not generated" in (
        window.dream_tab.output_box.toPlainText()
    )


def test_preview_runtime_bundle_opens_latest_written_script(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    window._last_written_dream_bundle = bundle

    opener = RuntimeBundleOpener(
        label="Fake Editor",
        stored_value="/Applications/FakeEditor.app",
        launch_target="/Applications/FakeEditor.app",
        launch_mode="mac_app",
    )
    launched: dict[str, object] = {}

    monkeypatch.setattr(
        window,
        "_available_runtime_bundle_openers",
        lambda: [opener],
    )
    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.QInputDialog.getItem",
        lambda *args, **kwargs: ("Fake Editor", True),
    )
    monkeypatch.setattr(
        window,
        "_launch_runtime_bundle_with_opener",
        lambda script_path, selected_opener: launched.update(
            {
                "path": str(script_path),
                "label": selected_opener.label,
                "stored_value": selected_opener.stored_value,
            }
        ),
    )

    window.preview_dream_runtime_bundle()

    assert launched["path"] == str(bundle.runtime_script_path)
    assert launched["label"] == "Fake Editor"
    assert "Opened DREAM runtime bundle preview" in (
        window.dream_tab.output_box.toPlainText()
    )
    reloaded_settings = SAXSProjectManager().load_project(project_dir)
    assert reloaded_settings.runtime_bundle_opener == opener.stored_value


def test_preview_runtime_bundle_reuses_saved_project_opener(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.runtime_bundle_opener = "/Applications/SavedEditor.app"
    manager.save_project(settings)

    window = SAXSMainWindow(initial_project_dir=project_dir)
    window._last_written_dream_bundle = bundle
    launched: dict[str, object] = {}

    monkeypatch.setattr(
        window,
        "_available_runtime_bundle_openers",
        lambda: [
            RuntimeBundleOpener(
                label="Saved Editor",
                stored_value="/Applications/SavedEditor.app",
                launch_target="/Applications/SavedEditor.app",
                launch_mode="mac_app",
            )
        ],
    )
    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.QInputDialog.getItem",
        lambda *args, **kwargs: pytest.fail(
            "The opener chooser should not appear for a saved project opener."
        ),
    )
    monkeypatch.setattr(
        window,
        "_launch_runtime_bundle_with_opener",
        lambda script_path, selected_opener: launched.update(
            {
                "path": str(script_path),
                "label": selected_opener.label,
                "stored_value": selected_opener.stored_value,
            }
        ),
    )

    window.preview_dream_runtime_bundle()

    assert launched["path"] == str(bundle.runtime_script_path)
    assert launched["stored_value"] == "/Applications/SavedEditor.app"


def test_run_dream_shows_progress_and_popup_can_be_closed(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    window._save_distribution_entries(entries)
    window.write_dream_bundle()

    def _fake_run_bundle(
        self,
        bundle,
        *,
        output_callback=None,
        output_interval_seconds=None,
    ):
        del output_interval_seconds
        if output_callback is not None:
            output_callback("DREAM sampler: initialization complete")
        time.sleep(0.15)
        if output_callback is not None:
            output_callback("DREAM sampler: collecting posterior samples")
        metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
        active_count = len(metadata["active_parameter_entries"])
        active_values = np.asarray(
            [
                float(entry["value"])
                for entry in metadata["active_parameter_entries"]
            ],
            dtype=float,
        )
        sampled_params = []
        log_ps = []
        for chain_index in range(2):
            chain_samples = []
            chain_logps = []
            for step_index in range(4):
                adjustment = (chain_index + 1) * (step_index + 1) * 0.01
                chain_samples.append(active_values[:active_count] + adjustment)
                chain_logps.append(-5.0 + chain_index + step_index * 0.5)
            sampled_params.append(chain_samples)
            log_ps.append(chain_logps)
        np.save(
            bundle.run_dir / "dream_sampled_params.npy",
            np.asarray(sampled_params, dtype=float),
        )
        np.save(
            bundle.run_dir / "dream_log_ps.npy",
            np.asarray(log_ps, dtype=float)[..., np.newaxis],
        )
        return {
            "sampled_params_path": str(
                bundle.run_dir / "dream_sampled_params.npy"
            ),
            "log_ps_path": str(bundle.run_dir / "dream_log_ps.npy"),
        }

    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.SAXSDreamWorkflow.run_bundle",
        _fake_run_bundle,
    )

    window.run_dream_bundle()
    QApplication.processEvents()

    assert window.dream_tab.progress_bar.minimum() == 0
    assert window.dream_tab.progress_bar.maximum() == 0
    assert not window.dream_tab.run_button.isEnabled()
    assert window._dream_progress_dialog is not None
    assert window._dream_progress_dialog.isVisible()

    deadline = time.time() + 2.0
    while (
        "DREAM sampler: initialization complete"
        not in window.dream_tab.output_box.toPlainText()
        and time.time() < deadline
    ):
        QApplication.processEvents()
        time.sleep(0.02)

    assert "DREAM Runtime Output" in window.dream_tab.output_box.toPlainText()
    assert (
        "DREAM sampler: initialization complete"
        in window.dream_tab.output_box.toPlainText()
    )

    window._dream_progress_dialog.close()
    QApplication.processEvents()
    assert not window._dream_progress_dialog.isVisible()

    deadline = time.time() + 5.0
    while window._dream_task_thread is not None and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)

    assert window._dream_task_thread is None
    assert window.dream_tab.run_button.isEnabled()
    assert window.dream_tab.progress_bar.maximum() == 1
    assert window.dream_tab.progress_bar.value() == 1
    assert (
        "DREAM sampler: collecting posterior samples"
        in window.dream_tab.output_box.toPlainText()
    )
    assert "DREAM run complete" in window.dream_tab.output_box.toPlainText()


def test_dream_results_loader_normalizes_singleton_logp_axis(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    bundle = _write_minimal_dream_results(project_dir)
    log_ps_path = bundle.run_dir / "dream_log_ps.npy"
    log_ps = np.load(log_ps_path)
    np.save(log_ps_path, np.asarray(log_ps, dtype=float)[..., np.newaxis])

    loader = SAXSDreamResultsLoader(bundle.run_dir, burnin_percent=0)
    summary = loader.get_summary()

    assert loader.log_ps.ndim == 2
    assert loader.log_ps.shape == (2, 4)
    assert summary.posterior_sample_count == 8


def test_dream_workflow_normalizes_stale_distribution_params_from_saved_map(
    qapp, tmp_path
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    workflow.parameter_map_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "structure": "PbI2",
                        "motif": "motif_A",
                        "param_type": "SAXS",
                        "param": "w0",
                        "value": 0.6,
                        "vary": True,
                        "distribution": "norm",
                        "dist_params": {
                            "loc": 0.6,
                            "scale": 0.2,
                            "s": 0.1,
                        },
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    entries = workflow.load_parameter_map()

    assert entries[0].distribution == "norm"
    assert entries[0].dist_params == {"loc": 0.6, "scale": 0.2}
    saved_project_map = json.loads(
        workflow.parameter_map_path.read_text(encoding="utf-8")
    )
    assert saved_project_map["entries"][0]["dist_params"] == {
        "loc": 0.6,
        "scale": 0.2,
    }

    bundle = workflow.create_runtime_bundle(entries=entries)
    saved_map = json.loads(
        bundle.parameter_map_path.read_text(encoding="utf-8")
    )
    assert saved_map["entries"][0]["dist_params"] == {
        "loc": 0.6,
        "scale": 0.2,
    }
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    assert metadata["active_parameter_entries"][0]["dist_params"] == {
        "loc": 0.6,
        "scale": 0.2,
    }


def test_dream_runtime_module_is_pickleable_for_parallel_execution(
    qapp, tmp_path
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    module_name, module, added_sys_path = workflow._load_runtime_module(bundle)
    try:
        pickled = pickle.dumps(module.active_log_likelihood)
    finally:
        workflow._unload_runtime_module(
            module_name,
            added_sys_path=added_sys_path,
        )

    assert pickled


def test_dream_runtime_module_saves_squeezed_log_ps(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    module_name, module, added_sys_path = workflow._load_runtime_module(bundle)
    try:
        monkeypatch.setattr(
            module,
            "run_dream",
            lambda **kwargs: (
                [
                    np.asarray(
                        [[0.6, 0.0, 0.05, 9.0, 0.0, 5e-4]], dtype=float
                    ),
                    np.asarray(
                        [[0.61, 0.0, 0.05, 9.1, 0.0, 5e-4]], dtype=float
                    ),
                ],
                [
                    np.asarray([[-5.0]], dtype=float),
                    np.asarray([[-4.8]], dtype=float),
                ],
            ),
        )
        module.run_sampler()
    finally:
        workflow._unload_runtime_module(
            module_name,
            added_sys_path=added_sys_path,
        )

    saved_log_ps = np.load(bundle.run_dir / "dream_log_ps.npy")
    assert saved_log_ps.shape == (2, 1)


def test_dream_runtime_module_handles_array_shaped_crossover_values(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    runtime_source = bundle.runtime_script_path.read_text(encoding="utf-8")
    assert "def _build_runtime_mp_context()" in runtime_source
    assert "mp_context=mp_context" in runtime_source

    module_name, module, added_sys_path = workflow._load_runtime_module(bundle)
    try:
        shared_vars = SimpleNamespace(
            cross_probs=mp.Array("d", [0.5, 0.5]),
            ncr_updates=mp.Array("d", [0.0, 0.0]),
            current_positions=mp.Array("d", [1.0, 2.0, 1.2, 2.4]),
            delta_m=mp.Array("d", [0.0, 0.0]),
            gamma_level_probs=mp.Array("d", [1.0]),
            ngamma_updates=mp.Array("d", [0.0]),
            delta_m_gamma=mp.Array("d", [0.0]),
        )
        monkeypatch.setattr(module, "Dream_shared_vars", shared_vars)
        module._PyDreamDream.estimate_crossover_probabilities.__globals__[
            "Dream_shared_vars"
        ] = shared_vars
        module._PyDreamDream.estimate_gamma_level_probs.__globals__[
            "Dream_shared_vars"
        ] = shared_vars

        fake_dream = SimpleNamespace(
            nCR=2,
            nchains=2,
            CR_values=np.asarray([0.5, 1.0], dtype=float),
            CR_probabilities=np.asarray([0.5, 0.5], dtype=float),
            ngamma=1,
            gamma_level_values=np.asarray([1], dtype=int),
        )

        crossover_value = module._PyDreamDream.set_CR(
            fake_dream,
            np.asarray([1.0, 0.0], dtype=float),
            np.asarray([0.5, 1.0], dtype=float),
        )
        assert float(crossover_value) == pytest.approx(0.5)

        cross_probs = module._PyDreamDream.estimate_crossover_probabilities(
            fake_dream,
            2,
            np.asarray([1.0, 2.0], dtype=float),
            np.asarray([1.1, 2.3], dtype=float),
            np.asarray([0.5], dtype=float),
        )
        assert np.asarray(cross_probs, dtype=float).shape == (2,)
        assert list(shared_vars.ncr_updates[:]) == [1.0, 0.0]

        gamma_probs = module._PyDreamDream.estimate_gamma_level_probs(
            fake_dream,
            2,
            np.asarray([1.0, 2.0], dtype=float),
            np.asarray([1.1, 2.3], dtype=float),
            np.asarray([1], dtype=int),
        )
        assert np.asarray(gamma_probs, dtype=float).shape == (1,)
        assert list(shared_vars.ngamma_updates[:]) == [1.0]
    finally:
        workflow._unload_runtime_module(
            module_name,
            added_sys_path=added_sys_path,
        )


def test_save_dream_settings_creates_named_preset_and_restores_active_state(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.dream_tab.chains_spin.setValue(6)
    window.dream_tab.iterations_spin.setValue(2500)
    window.dream_tab.posterior_filter_combo.setCurrentIndex(1)
    window.dream_tab.posterior_top_percent_spin.setValue(12.5)
    window.dream_tab.credible_interval_low_spin.setValue(10.0)
    window.dream_tab.credible_interval_high_spin.setValue(90.0)
    window.dream_tab.violin_sample_source_combo.setCurrentIndex(1)

    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("Preset A", True),
    )

    window.save_dream_settings()

    preset_index = window.dream_tab.settings_preset_combo.findText("Preset A")
    assert preset_index >= 0
    assert (
        window.dream_tab.settings_preset_combo.currentText()
        == window.dream_tab.ACTIVE_SETTINGS_LABEL
    )

    window.dream_tab.chains_spin.setValue(11)
    window.dream_tab.iterations_spin.setValue(3333)
    window.dream_tab.posterior_filter_combo.setCurrentIndex(2)
    window.dream_tab.posterior_top_n_spin.setValue(7)
    window.dream_tab.violin_sample_source_combo.setCurrentIndex(0)

    window.dream_tab.settings_preset_combo.setCurrentIndex(preset_index)
    QApplication.processEvents()

    assert window.dream_tab.chains_spin.value() == 6
    assert window.dream_tab.iterations_spin.value() == 2500
    assert (
        window.dream_tab.posterior_filter_combo.currentData()
        == "top_percent_logp"
    )
    assert window.dream_tab.posterior_top_percent_spin.value() == 12.5
    assert window.dream_tab.credible_interval_low_spin.value() == 10.0
    assert window.dream_tab.credible_interval_high_spin.value() == 90.0
    assert (
        window.dream_tab.violin_sample_source_combo.currentData()
        == "map_chain_only"
    )

    active_index = window.dream_tab.settings_preset_combo.findText(
        window.dream_tab.ACTIVE_SETTINGS_LABEL
    )
    window.dream_tab.settings_preset_combo.setCurrentIndex(active_index)
    QApplication.processEvents()

    assert window.dream_tab.chains_spin.value() == 11
    assert window.dream_tab.iterations_spin.value() == 3333
    assert (
        window.dream_tab.posterior_filter_combo.currentData() == "top_n_logp"
    )
    assert window.dream_tab.posterior_top_n_spin.value() == 7
    assert (
        window.dream_tab.violin_sample_source_combo.currentData()
        == "filtered_posterior"
    )


def test_dream_posterior_filter_controls_enable_only_relevant_inputs(
    qapp, tmp_path
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.dream_tab.posterior_filter_combo.setCurrentIndex(1)
    QApplication.processEvents()

    assert window.dream_tab.posterior_top_percent_spin.isEnabled()
    assert not window.dream_tab.posterior_top_n_spin.isEnabled()

    window.dream_tab.posterior_filter_combo.setCurrentIndex(2)
    QApplication.processEvents()

    assert not window.dream_tab.posterior_top_percent_spin.isEnabled()
    assert window.dream_tab.posterior_top_n_spin.isEnabled()


def test_load_latest_dream_results_updates_both_plot_panels(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    _write_minimal_dream_results(project_dir)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    window.dream_tab.bestfit_method_combo.setCurrentIndex(1)
    window.dream_tab.violin_mode_combo.setCurrentIndex(2)
    window.dream_tab.posterior_filter_combo.setCurrentIndex(2)
    window.dream_tab.posterior_top_n_spin.setValue(1)
    window.dream_tab.credible_interval_low_spin.setValue(5.0)
    window.dream_tab.credible_interval_high_spin.setValue(95.0)
    window.dream_tab.violin_sample_source_combo.setCurrentIndex(1)

    window.load_latest_results()

    assert "Best-fit method: chain_mean" in (
        window.dream_tab.output_box.toPlainText()
    )
    assert "Posterior filter: top_n_logp" in (
        window.dream_tab.output_box.toPlainText()
    )
    assert "Posterior samples kept: 1" in (
        window.dream_tab.output_box.toPlainText()
    )
    assert "Violin data mode: weights_only" in (
        window.dream_tab.output_box.toPlainText()
    )
    assert "Violin sample source: map_chain_only" in (
        window.dream_tab.output_box.toPlainText()
    )
    assert "p5=" in window.dream_tab.output_box.toPlainText()
    assert "p95=" in window.dream_tab.output_box.toPlainText()
    assert (
        window.dream_tab.model_figure.axes[0]
        .get_title()
        .startswith("DREAM refinement:")
    )
    metric_text = "\n".join(
        text.get_text() for text in window.dream_tab.model_figure.axes[0].texts
    )
    assert "RMSE:" in metric_text
    assert "Mean |res|:" in metric_text
    assert "R²:" in metric_text
    assert (
        window.dream_tab.violin_figure.axes[0].get_title()
        == "Posterior parameter distributions"
    )
    tick_labels = [
        label.get_text()
        for label in window.dream_tab.violin_figure.axes[0].get_xticklabels()
    ]
    assert "w0 (A)" in tick_labels


def test_dream_model_metrics_box_updates_with_bestfit_method(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    _write_minimal_dream_results(project_dir)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.load_latest_results()
    axis = window.dream_tab.model_figure.axes[0]
    first_metrics = "\n".join(text.get_text() for text in axis.texts)

    window.dream_tab.bestfit_method_combo.setCurrentIndex(2)
    QApplication.processEvents()

    axis = window.dream_tab.model_figure.axes[0]
    second_metrics = "\n".join(text.get_text() for text in axis.texts)

    assert "RMSE:" in second_metrics
    assert "Mean |res|:" in second_metrics
    assert "R²:" in second_metrics
    assert first_metrics != second_metrics


def test_dream_violin_scale_modes_and_palette_controls(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    _write_minimal_dream_results(project_dir)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    palette_index = window.dream_tab.violin_palette_combo.findData("plasma")
    window.load_latest_results()
    window.dream_tab.violin_value_scale_combo.setCurrentIndex(1)
    window.dream_tab.violin_palette_combo.setCurrentIndex(palette_index)
    window.dream_tab._configure_plot_color_button(
        window.dream_tab.violin_point_color_button,
        "tab:blue",
        label="Point",
    )
    window.dream_tab.visualization_settings_changed.emit()
    QApplication.processEvents()

    axis = window.dream_tab.violin_figure.axes[0]
    tick_labels = [label.get_text() for label in axis.get_xticklabels()]
    assert tick_labels == ["w0 (A)"]
    assert axis.get_ylabel() == "Weight fraction"
    assert axis.get_title() == "Posterior weight distributions"
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    body = next(
        collection
        for collection in axis.collections
        if isinstance(collection, PolyCollection)
    )
    assert to_hex(body.get_facecolor()[0], keep_alpha=False) == to_hex(
        colormaps["plasma"](0.6),
        keep_alpha=False,
    )
    assert to_hex(
        axis.collections[-1].get_facecolor()[0], keep_alpha=False
    ) == to_hex(
        "tab:blue",
        keep_alpha=False,
    )

    window.dream_tab.violin_value_scale_combo.setCurrentIndex(2)
    QApplication.processEvents()

    axis = window.dream_tab.violin_figure.axes[0]
    assert axis.get_ylabel() == "Normalized parameter value"
    assert axis.get_title() == "Posterior parameter distributions (normalized)"
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    normalized_labels = [
        label.get_text()
        for label in axis.get_xticklabels()
        if label.get_text()
    ]
    assert "w0 (A)" in normalized_labels
    assert "solv_w" in normalized_labels


def test_dream_violin_custom_color_controls_apply_to_plot(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    _write_minimal_dream_results(project_dir)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    chosen_colors = iter(
        [
            QColor("#123456"),
            QColor("#fedcba"),
            QColor("#654321"),
            QColor("#abcdef"),
        ]
    )
    monkeypatch.setattr(
        "saxshell.saxs.ui.dream_tab.QColorDialog.getColor",
        lambda *args, **kwargs: next(chosen_colors),
    )

    palette_index = window.dream_tab.violin_palette_combo.findData(
        "custom_solid"
    )
    window.load_latest_results()
    window.dream_tab.violin_palette_combo.setCurrentIndex(palette_index)
    window.dream_tab._choose_violin_custom_color()
    window.dream_tab._choose_violin_point_color()
    window.dream_tab._choose_interval_color()
    window.dream_tab._choose_median_color()
    QApplication.processEvents()

    axis = window.dream_tab.violin_figure.axes[0]
    body = next(
        collection
        for collection in axis.collections
        if isinstance(collection, PolyCollection)
    )
    assert to_hex(body.get_facecolor()[0], keep_alpha=False) == "#123456"
    assert (
        to_hex(
            axis.collections[-1].get_facecolor()[0],
            keep_alpha=False,
        )
        == "#fedcba"
    )
    line_colors = [
        to_hex(color, keep_alpha=False)
        for collection in axis.collections
        if isinstance(collection, LineCollection)
        for color in collection.get_colors()
    ]
    assert "#654321" in line_colors
    assert "#abcdef" in line_colors


def test_dream_results_loader_can_order_weight_violin_labels_by_structure(
    tmp_path,
):
    run_dir = _write_weight_order_dream_results(tmp_path)
    loader = SAXSDreamResultsLoader(run_dir, burnin_percent=0)

    weight_index_plot = loader.build_violin_data(
        mode="weights_only",
        weight_order="weight_index",
    )
    structure_order_plot = loader.build_violin_data(
        mode="weights_only",
        weight_order="structure_order",
    )

    assert weight_index_plot.parameter_names == ["w2", "w0", "w1"]
    assert weight_index_plot.display_names == [
        "w2 (PbI2O)",
        "w0 (I2)",
        "w1 (Pb2)",
    ]
    assert structure_order_plot.parameter_names == ["w0", "w1", "w2"]
    assert structure_order_plot.display_names == [
        "w0 (I2)",
        "w1 (Pb2)",
        "w2 (PbI2O)",
    ]


def test_dream_results_loader_filters_posterior_samples(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    bundle = _write_minimal_dream_results(project_dir)
    loader = SAXSDreamResultsLoader(bundle.run_dir, burnin_percent=0)

    summary = loader.get_summary(
        bestfit_method="median",
        posterior_filter_mode="top_n_logp",
        posterior_top_n=1,
        credible_interval_low=10.0,
        credible_interval_high=90.0,
    )
    violin_plot = loader.build_violin_data(
        mode="varying_parameters",
        posterior_filter_mode="top_percent_logp",
        posterior_top_percent=25.0,
        sample_source="map_chain_only",
    )

    assert summary.posterior_filter_mode == "top_n_logp"
    assert summary.posterior_sample_count == 1
    assert summary.credible_interval_low == 10.0
    assert summary.credible_interval_high == 90.0
    assert np.allclose(
        summary.interval_low_values,
        summary.interval_high_values,
    )
    assert np.allclose(
        summary.bestfit_params,
        summary.interval_low_values,
    )
    assert violin_plot.sample_source == "map_chain_only"
    assert violin_plot.sample_count == 2


def test_dream_runtime_bundle_carries_selected_solvent_trace(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    solvent_q = np.linspace(0.05, 0.3, 8)
    solvent_intensity = np.linspace(3.0, 4.4, 8)
    solvent_path = tmp_path / "solvent_reference_trace.dat"
    np.savetxt(solvent_path, np.column_stack([solvent_q, solvent_intensity]))

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.solvent_data_path = str(solvent_path)
    settings.copied_solvent_data_file = None
    manager.save_project(settings)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))

    assert np.allclose(metadata["solvent_intensities"], solvent_intensity)


def test_dream_plot_data_exports_save_into_project_plots(qapp, tmp_path):
    del qapp
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    _write_minimal_dream_results(project_dir)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    class _AcceptedDialog:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.selected_options = SimpleNamespace(
                output_dir=paths.plots_dir,
                base_name="dream_violin_export_test",
                save_csv=True,
                save_pkl=True,
            )

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.DreamViolinExportDialog",
        _AcceptedDialog,
    )

    window.load_latest_results()
    window.dream_tab.violin_mode_combo.setCurrentIndex(2)
    window.save_dream_model_fit()
    window.save_dream_violin_data()
    monkeypatch.undo()

    model_exports = sorted(paths.plots_dir.glob("dream_model_fit_*.csv"))
    violin_csv_exports = sorted(paths.plots_dir.glob("dream_violin_*.csv"))
    violin_pkl_exports = sorted(paths.plots_dir.glob("dream_violin_*.pkl"))
    dialog_csv_export = paths.plots_dir / "dream_violin_export_test.csv"
    dialog_pkl_export = paths.plots_dir / "dream_violin_export_test.pkl"

    assert model_exports
    assert violin_csv_exports
    assert violin_pkl_exports
    assert dialog_csv_export.is_file()
    assert dialog_pkl_export.is_file()
    assert "q,experimental_intensity,model_intensity" in model_exports[
        -1
    ].read_text(encoding="utf-8")
    assert "w0 (A)" in dialog_csv_export.read_text(encoding="utf-8")
    violin_payload = pickle.loads(dialog_pkl_export.read_bytes())
    assert violin_payload["violin_plot"]["display_names"] == ["w0 (A)"]
    assert violin_payload["plot_payload"]["ylabel"] == "Parameter value"
    assert np.asarray(violin_payload["plot_payload"]["samples"]).shape[1] == 1


def test_prefit_template_change_warning_can_be_cancelled(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    if window.prefit_tab.template_combo.count() < 2:
        pytest.skip("Need at least two templates to test template switching.")

    original_template = window.prefit_workflow.template_spec.name
    alternative = next(
        str(window.prefit_tab.template_combo.itemData(index) or "")
        for index in range(window.prefit_tab.template_combo.count())
        if str(window.prefit_tab.template_combo.itemData(index) or "")
        != original_template
    )

    monkeypatch.setattr(
        window,
        "_confirm_prefit_template_change",
        lambda current, new: (False, False),
    )

    window.prefit_tab.set_selected_template(alternative, emit_signal=True)

    assert window.prefit_workflow.template_spec.name == original_template
    assert window.prefit_tab.selected_template_name() == original_template


def test_prefit_template_change_can_switch_back_to_original(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    if window.prefit_tab.template_combo.count() < 2:
        pytest.skip("Need at least two templates to test template switching.")

    original_template = window.prefit_workflow.template_spec.name
    alternative = next(
        str(window.prefit_tab.template_combo.itemData(index) or "")
        for index in range(window.prefit_tab.template_combo.count())
        if str(window.prefit_tab.template_combo.itemData(index) or "")
        != original_template
    )

    monkeypatch.setattr(
        window,
        "_confirm_prefit_template_change",
        lambda current, new: (True, False),
    )

    window.prefit_tab.set_selected_template(alternative, emit_signal=True)
    assert window.prefit_workflow.template_spec.name == alternative

    window.prefit_tab.set_selected_template(
        original_template, emit_signal=True
    )

    assert window.prefit_workflow.template_spec.name == original_template
    assert window.prefit_tab.selected_template_name() == original_template


def test_project_setup_inputs_stay_locked_until_project_selected(
    qapp, tmp_path
):
    del qapp
    window = SAXSMainWindow()

    assert not window.project_setup_tab.forward_model_group.isEnabled()
    assert not window.project_setup_tab.model_group.isEnabled()
    assert not window.project_setup_tab.prior_mode_combo.isEnabled()
    assert not window.project_setup_tab.generate_prior_plot_button.isEnabled()
    assert window.prefit_tab.template_combo.count() >= 1
    assert window.project_setup_tab.use_experimental_grid()
    assert not window.project_setup_tab.resample_points_spin.isEnabled()

    project_dir = tmp_path / "Created SAXS Project"
    window.project_setup_tab.project_name_edit.setText("Created SAXS Project")
    window.project_setup_tab.project_dir_edit.setText(str(tmp_path))
    window.create_project_from_tab()

    assert window.project_setup_tab.forward_model_group.isEnabled()
    assert window.project_setup_tab.model_group.isEnabled()
    assert window.project_setup_tab.prior_mode_combo.isEnabled()
    assert window.project_setup_tab.generate_prior_plot_button.isEnabled()
    assert window.project_setup_tab.project_dir_edit.text() == str(tmp_path)
    assert window.project_setup_tab.project_name_edit.text() == (
        "Created SAXS Project"
    )
    assert window.project_setup_tab.open_project_dir_edit.text().endswith(
        "Created SAXS Project"
    )
    assert window.current_settings.project_name == "Created SAXS Project"
    assert Path(window.current_settings.project_dir) == project_dir.resolve()

    window.project_setup_tab.use_experimental_grid_checkbox.setChecked(False)

    assert window.project_setup_tab.resample_points_spin.isEnabled()


def test_recognized_cluster_table_is_resizable_scrollable_and_saves_colors(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)
    window.project_setup_tab.apply_cluster_import_data(
        ["A"],
        [
            {
                "structure": "A",
                "motif": "no_motif",
                "count": 1,
                "weight": 1.0,
                "atom_fraction_percent": 100.0,
                "structure_fraction_percent": 100.0,
            }
        ],
    )

    table = window.project_setup_tab.recognized_clusters_table
    assert (
        table.horizontalHeader().sectionResizeMode(0)
        == QHeaderView.ResizeMode.Interactive
    )
    assert (
        table.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    monkeypatch.setattr(
        "saxshell.saxs.ui.project_setup_tab.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )

    window.project_setup_tab._on_recognized_cluster_cell_clicked(0, 7)

    assert table.item(0, 7).text() == "#123456"
    assert window.project_setup_tab.component_trace_colors() == {
        "A_no_motif": "#123456"
    }

    window.save_project_state()

    settings = SAXSProjectManager().load_project(project_dir)
    assert settings.component_trace_colors == {"A_no_motif": "#123456"}


def test_open_project_uses_existing_project_field(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow()

    window.project_setup_tab.open_project_dir_edit.setText(str(project_dir))
    window.open_project_from_dialog()

    assert window.current_settings is not None
    assert window.current_settings.project_dir == str(project_dir.resolve())
    assert window.project_setup_tab.project_name_edit.text() == "saxs_project"


def test_loaded_project_prefers_original_experimental_reference_in_ui(
    qapp, tmp_path
):
    del qapp
    manager = SAXSProjectManager()
    source_dir = tmp_path / "source_data"
    source_dir.mkdir()
    source_path = source_dir / "exp_source.txt"
    np.savetxt(
        source_path,
        np.column_stack(
            [
                np.asarray([0.05, 0.10], dtype=float),
                np.asarray([10.0, 9.0], dtype=float),
            ]
        ),
    )

    project_dir = tmp_path / "project_with_copy"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)
    copied_path = paths.experimental_data_dir / source_path.name
    copied_path.write_text(
        source_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    settings.experimental_data_path = str(source_path)
    settings.copied_experimental_data_file = str(copied_path)
    manager.save_project(settings)

    window = SAXSMainWindow(initial_project_dir=project_dir)

    assert window.project_setup_tab.experimental_data_edit.text() == str(
        source_path
    )


def test_prefit_recommended_scale_button_updates_scale_bounds(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    scale_row = window.prefit_tab.find_parameter_row("scale")
    assert scale_row >= 0
    window.prefit_tab.parameter_table.item(scale_row, 3).setText("1e-6")
    window.prefit_tab.parameter_table.item(scale_row, 5).setText("1e-7")
    window.prefit_tab.parameter_table.item(scale_row, 6).setText("1e-5")

    window.apply_recommended_scale_settings()

    scale_entry = next(
        entry
        for entry in window.prefit_tab.parameter_entries()
        if entry.name == "scale"
    )
    assert scale_entry.vary
    assert scale_entry.value == pytest.approx(5e-4)
    assert scale_entry.minimum == pytest.approx(5e-5)
    assert scale_entry.maximum == pytest.approx(5e-3)
    assert "Applied recommended scale settings." in (
        window.prefit_tab.output_box.toPlainText()
    )


def test_best_prefit_preset_saves_resets_and_reloads(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.prefit_tab.set_parameter_row(
        "scale",
        value=7e-4,
        minimum=1e-5,
        maximum=8e-3,
        vary=True,
    )
    window.prefit_tab.set_parameter_row("offset", value=0.125)
    window.set_best_prefit_parameters()

    settings = SAXSProjectManager().load_project(project_dir)
    assert settings.best_prefit_template == (
        "template_pd_likelihood_monosq_decoupled"
    )
    assert settings.template_reset_template == (
        "template_pd_likelihood_monosq_decoupled"
    )
    assert settings.best_prefit_parameter_entries
    assert settings.template_reset_parameter_entries

    window.prefit_tab.set_parameter_row("scale", value=2e-3)
    window.prefit_tab.set_parameter_row("offset", value=0.333)
    window.reset_parameters_to_best_prefit()

    best_entries = {
        entry.name: entry for entry in window.prefit_tab.parameter_entries()
    }
    assert best_entries["scale"].value == pytest.approx(7e-4)
    assert best_entries["offset"].value == pytest.approx(0.125)

    window.prefit_tab.set_parameter_row("scale", value=2e-3)
    window.reset_prefit_entries()

    template_entries = {
        entry.name: entry for entry in window.prefit_tab.parameter_entries()
    }
    assert template_entries["scale"].value == pytest.approx(5e-4)
    assert template_entries["offset"].value == pytest.approx(0.0)

    reloaded_window = SAXSMainWindow(initial_project_dir=project_dir)
    reloaded_entries = {
        entry.name: entry
        for entry in reloaded_window.prefit_tab.parameter_entries()
    }
    assert reloaded_entries["scale"].value == pytest.approx(7e-4)
    assert reloaded_entries["offset"].value == pytest.approx(0.125)


def test_restore_prefit_state_recovers_saved_parameters_and_run_config(
    qapp, tmp_path
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.prefit_tab.set_parameter_row(
        "scale",
        value=7e-4,
        minimum=1e-5,
        maximum=8e-3,
        vary=True,
    )
    window.prefit_tab.set_parameter_row("offset", value=0.125)
    window.prefit_tab.set_run_config(method="powell", max_nfev=4321)
    window.prefit_tab.set_autosave(True)
    window.set_best_prefit_parameters()
    window.save_prefit()

    assert window.prefit_tab.restore_state_button.isEnabled()
    saved_state_name = window.prefit_tab.selected_saved_state_name()
    assert saved_state_name is not None
    assert saved_state_name.startswith("prefit_")

    window.prefit_tab.set_parameter_row("scale", value=2e-3)
    window.prefit_tab.set_parameter_row("offset", value=0.333)
    window.prefit_tab.set_run_config(method="nelder", max_nfev=9999)
    window.prefit_tab.set_autosave(False)
    window.restore_prefit_state()

    restored_entries = {
        entry.name: entry for entry in window.prefit_tab.parameter_entries()
    }
    assert restored_entries["scale"].value == pytest.approx(7e-4)
    assert restored_entries["offset"].value == pytest.approx(0.125)
    assert window.prefit_tab.run_config().method == "powell"
    assert window.prefit_tab.run_config().max_nfev == 4321
    assert window.prefit_tab.autosave_checkbox.isChecked()

    window.prefit_tab.set_parameter_row("scale", value=1.5e-3)
    window.reset_parameters_to_best_prefit()
    best_entries = {
        entry.name: entry for entry in window.prefit_tab.parameter_entries()
    }
    assert best_entries["scale"].value == pytest.approx(7e-4)


def test_generate_prior_weights_does_not_force_prefit_without_components(
    qapp, tmp_path, monkeypatch
):
    del qapp
    window = SAXSMainWindow()
    window.project_setup_tab.project_name_edit.setText("demo_project")
    window.project_setup_tab.project_dir_edit.setText(str(tmp_path))
    window.create_project_from_tab()

    scheduled: dict[str, object] = {}

    monkeypatch.setattr(
        window,
        "_start_project_task",
        lambda task_name, task_fn, start_message, settings=None: scheduled.update(
            {
                "task_name": task_name,
                "task_fn": task_fn,
                "start_message": start_message,
                "settings": settings,
            }
        ),
    )
    saved_paths: list[Path] = []
    monkeypatch.setattr(
        window.project_manager,
        "save_project",
        lambda settings: saved_paths.append(
            build_project_paths(settings.project_dir).project_file
        )
        or build_project_paths(settings.project_dir).project_file,
    )

    window.build_prior_weights()

    assert scheduled["task_name"] == "build_prior_weights"
    assert scheduled["start_message"] == "Generating prior weights..."
    assert scheduled["settings"] is window.current_settings
    assert saved_paths


def test_generate_plot_opens_standalone_prior_histogram_window(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.show_prior_histogram_window()

    prior_window = window._prior_histogram_windows[-1]
    assert isinstance(prior_window, PriorHistogramWindow)
    assert prior_window.parent() is None
    assert prior_window.isWindow()
    assert prior_window.toolbar is not None


def test_solvent_sort_mode_populates_secondary_atom_filter(qapp):
    del qapp
    tab = ProjectSetupTab()
    tab.set_project_selected(True)
    tab.apply_cluster_import_data(
        ["Pb", "I", "O"],
        [
            {
                "structure": "PbI2",
                "motif": "motif_A",
                "count": 2,
                "weight": 0.4,
                "atom_fraction_percent": 40.0,
                "structure_fraction_percent": 40.0,
            },
            {
                "structure": "Pb2I4",
                "motif": "motif_B",
                "count": 3,
                "weight": 0.6,
                "atom_fraction_percent": 60.0,
                "structure_fraction_percent": 60.0,
            },
        ],
    )

    tab.prior_mode_combo.setCurrentText("Solvent Sort - Structure Fraction")

    assert not tab.secondary_filter_combo.isHidden()
    assert tab.secondary_filter_combo.isEnabled()
    assert tab.secondary_filter_combo.count() == 1
    assert tab.secondary_filter_combo.currentText() == "O"


def test_save_prior_plot_png_writes_to_project_plots_dir(qapp, tmp_path):
    del qapp
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    window.save_prior_plot_png()

    saved_images = list(paths.plots_dir.glob("prior_histogram_*.png"))
    assert saved_images


def test_prior_histogram_window_uses_secondary_filter_and_colormap(
    qapp, tmp_path
):
    del qapp
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 5,
                "available_elements": ["Pb", "I", "O"],
                "structures": {
                    "PbI2": {
                        "motif_A": {
                            "count": 2,
                            "weight": 0.4,
                            "profile_file": "A_no_motif.txt",
                            "secondary_atom_distributions": {
                                "O": {"0": 1, "1": 1}
                            },
                        }
                    },
                    "Pb2I4": {
                        "motif_B": {
                            "count": 3,
                            "weight": 0.6,
                            "profile_file": "A_no_motif.txt",
                            "secondary_atom_distributions": {
                                "O": {"0": 1, "2": 2}
                            },
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    window = SAXSMainWindow(initial_project_dir=project_dir)
    window.project_setup_tab.apply_cluster_import_data(
        ["Pb", "I", "O"],
        [
            {
                "structure": "PbI2",
                "motif": "motif_A",
                "count": 2,
                "weight": 0.4,
                "atom_fraction_percent": 40.0,
                "structure_fraction_percent": 40.0,
            },
            {
                "structure": "Pb2I4",
                "motif": "motif_B",
                "count": 3,
                "weight": 0.6,
                "atom_fraction_percent": 60.0,
                "structure_fraction_percent": 60.0,
            },
        ],
    )
    window.project_setup_tab.prior_mode_combo.setCurrentText(
        "Solvent Sort - Structure Fraction"
    )
    window.project_setup_tab.prior_color_combo.setCurrentText("viridis")

    window.show_prior_histogram_window()

    prior_window = window._prior_histogram_windows[-1]
    assert prior_window.mode == "solvent_sort_structure_fraction"
    assert prior_window.secondary_element == "O"
    assert prior_window.cmap == "viridis"


def test_create_project_warns_before_overwriting_existing_folder(
    qapp, tmp_path, monkeypatch
):
    del qapp
    existing_project_dir = tmp_path / "existing_project"
    existing_project_dir.mkdir()
    window = SAXSMainWindow()
    window.project_setup_tab.project_name_edit.setText("existing_project")
    window.project_setup_tab.project_dir_edit.setText(str(tmp_path))

    monkeypatch.setattr(
        "saxshell.saxs.ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )

    window.create_project_from_tab()

    assert window.current_settings is None
    assert not (existing_project_dir / "saxs_project.json").exists()


def test_selecting_clusters_directory_triggers_project_autosave(
    qapp, tmp_path, monkeypatch
):
    del qapp
    window = SAXSMainWindow()
    window.project_setup_tab.project_name_edit.setText("demo_project")
    window.project_setup_tab.project_dir_edit.setText(str(tmp_path))
    window.create_project_from_tab()

    clusters_dir = tmp_path / "clusters"
    (clusters_dir / "PbI2").mkdir(parents=True)
    (clusters_dir / "PbI2" / "frame_0001.xyz").write_text(
        "2\ncomment\nPb 0.0 0.0 0.0\nI 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "saxshell.saxs.ui.project_setup_tab.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(clusters_dir),
    )
    monkeypatch.setattr(
        window, "_start_project_task", lambda *args, **kwargs: None
    )
    saved_clusters: list[str | None] = []
    monkeypatch.setattr(
        window.project_manager,
        "save_project",
        lambda settings: saved_clusters.append(settings.clusters_dir)
        or build_project_paths(settings.project_dir).project_file,
    )

    window.project_setup_tab._choose_clusters_directory()

    assert saved_clusters[-1] == str(clusters_dir)


def test_selecting_experimental_file_triggers_project_autosave(
    qapp, tmp_path, monkeypatch
):
    del qapp
    window = SAXSMainWindow()
    window.project_setup_tab.project_name_edit.setText("demo_project")
    window.project_setup_tab.project_dir_edit.setText(str(tmp_path))
    window.create_project_from_tab()

    data_path = tmp_path / "exp_demo.txt"
    np.savetxt(
        data_path,
        np.column_stack(
            [
                np.asarray([0.05, 0.10], dtype=float),
                np.asarray([10.0, 9.5], dtype=float),
            ]
        ),
    )
    monkeypatch.setattr(
        "saxshell.saxs.ui.project_setup_tab.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (
            str(data_path),
            "Data files (*.txt *.dat *.iq)",
        ),
    )
    saved_data_paths: list[str | None] = []
    monkeypatch.setattr(
        window.project_manager,
        "save_project",
        lambda settings: saved_data_paths.append(
            settings.experimental_data_path
        )
        or build_project_paths(settings.project_dir).project_file,
    )

    window.project_setup_tab._choose_experimental_file()

    assert saved_data_paths[-1] == str(data_path)


def test_experimental_data_header_dialog_recovers_header_rows(qapp, tmp_path):
    del qapp
    data_path = tmp_path / "exp_with_header.txt"
    data_path.write_text(
        "Example SAXS export\n"
        "q intensity sigma\n"
        "0.05 10.0 0.1\n"
        "0.10 9.5 0.1\n",
        encoding="utf-8",
    )

    dialog = ExperimentalDataHeaderDialog(data_path)
    dialog.header_rows_spin.setValue(2)
    dialog._try_accept()

    assert dialog.accepted_summary is not None
    assert dialog.header_rows() == 2
    assert dialog.accepted_summary.header_rows == 2
    assert np.allclose(dialog.accepted_summary.q_values, [0.05, 0.10])
    assert np.allclose(
        dialog.accepted_summary.intensities,
        [10.0, 9.5],
    )


def test_load_experimental_data_file_detects_three_column_headers(
    tmp_path,
):
    data_path = tmp_path / "exp_three_columns.txt"
    data_path.write_text(
        "q_demo\tintensity_demo\terror_demo\n"
        "0.05\t10.0\t0.1\n"
        "0.10\t9.5\t0.2\n",
        encoding="utf-8",
    )

    summary = load_experimental_data_file(data_path)

    assert summary.header_rows == 1
    assert summary.column_names == [
        "q_demo",
        "intensity_demo",
        "error_demo",
    ]
    assert summary.q_column == 0
    assert summary.intensity_column == 1
    assert summary.error_column == 2
    assert summary.errors is not None
    assert np.allclose(summary.q_values, [0.05, 0.10])
    assert np.allclose(summary.intensities, [10.0, 9.5])
    assert np.allclose(summary.errors, [0.1, 0.2])


def test_experimental_data_header_dialog_allows_manual_column_selection(
    qapp, tmp_path
):
    del qapp
    data_path = tmp_path / "exp_swapped_columns.txt"
    data_path.write_text(
        "intensity_value\tq_value\tsigma_value\n"
        "10.0\t0.05\t0.1\n"
        "9.5\t0.10\t0.2\n",
        encoding="utf-8",
    )

    dialog = ExperimentalDataHeaderDialog(data_path)
    dialog.header_rows_spin.setValue(1)
    dialog.q_column_combo.setCurrentIndex(dialog.q_column_combo.findData(1))
    dialog.intensity_column_combo.setCurrentIndex(
        dialog.intensity_column_combo.findData(0)
    )
    dialog.error_column_combo.setCurrentIndex(
        dialog.error_column_combo.findData(2)
    )
    dialog._try_accept()

    assert dialog.accepted_summary is not None
    assert dialog.accepted_summary.column_names == [
        "intensity_value",
        "q_value",
        "sigma_value",
    ]
    assert dialog.q_column() == 1
    assert dialog.intensity_column() == 0
    assert dialog.error_column() == 2
    assert np.allclose(dialog.accepted_summary.q_values, [0.05, 0.10])
    assert np.allclose(
        dialog.accepted_summary.intensities,
        [10.0, 9.5],
    )
    assert np.allclose(dialog.accepted_summary.errors, [0.1, 0.2])


def test_project_setup_preview_updates_with_experimental_q_range(
    qapp, tmp_path
):
    del qapp
    data_path = tmp_path / "exp_preview.txt"
    np.savetxt(
        data_path,
        np.column_stack(
            [
                np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float),
                np.asarray([100.0, 80.0, 55.0, 30.0], dtype=float),
            ]
        ),
    )

    summary = load_experimental_data_file(data_path)
    tab = ProjectSetupTab()
    tab._apply_experimental_file(data_path, summary)
    tab.draw_component_plot(None)

    preview_axis = tab.component_figure.axes[0]
    assert tab.component_log_x_checkbox.isChecked()
    assert tab.component_log_y_checkbox.isChecked()
    assert preview_axis.get_title() == "Experimental Data Preview"
    assert preview_axis.get_xlabel() == "q (Å⁻¹)"
    assert preview_axis.get_ylabel() == "Intensity (arb. units)"
    assert preview_axis.get_xscale() == "log"
    assert preview_axis.get_yscale() == "log"

    tab.qmin_edit.setText("0.08")
    tab.qmax_edit.setText("0.12")
    tab.component_log_x_checkbox.setChecked(False)
    tab.draw_component_plot(None)

    preview_axis = tab.component_figure.axes[0]
    legend_labels = preview_axis.get_legend_handles_labels()[1]
    assert "Selected q-range" in legend_labels
    assert preview_axis.get_xscale() == "linear"
    assert preview_axis.get_yscale() == "log"


def test_project_setup_preview_plots_solvent_data_in_green(qapp, tmp_path):
    del qapp
    experimental_path = tmp_path / "exp_preview.txt"
    solvent_path = tmp_path / "solvent_preview.txt"
    q_values = np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float)
    np.savetxt(
        experimental_path,
        np.column_stack(
            [q_values, np.asarray([100.0, 80.0, 55.0, 30.0], dtype=float)]
        ),
    )
    np.savetxt(
        solvent_path,
        np.column_stack(
            [q_values, np.asarray([12.0, 11.0, 10.0, 9.0], dtype=float)]
        ),
    )

    experimental_summary = load_experimental_data_file(experimental_path)
    solvent_summary = load_experimental_data_file(solvent_path)
    tab = ProjectSetupTab()
    tab._apply_experimental_file(experimental_path, experimental_summary)
    tab._apply_solvent_file(solvent_path, solvent_summary)

    preview_axis = tab.component_figure.axes[0]
    labels = [line.get_label() for line in preview_axis.get_lines()]

    assert "Experimental data" in labels
    assert "Solvent data" in labels
    solvent_line = next(
        line
        for line in preview_axis.get_lines()
        if line.get_label() == "Solvent data"
    )
    assert to_hex(solvent_line.get_color()) == "#008000"
    assert preview_axis.get_legend() is not None


def test_project_setup_data_trace_controls_toggle_and_persist(
    qapp, tmp_path, monkeypatch
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    solvent_path = tmp_path / "solvent_control_trace.txt"
    q_values = np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float)
    np.savetxt(
        solvent_path,
        np.column_stack(
            [q_values, np.asarray([12.0, 11.0, 10.0, 9.0], dtype=float)]
        ),
    )

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.solvent_data_path = str(solvent_path)
    settings.copied_solvent_data_file = None
    manager.save_project(settings)

    window = SAXSMainWindow(initial_project_dir=project_dir)
    tab = window.project_setup_tab

    def _pick_color(*args, **kwargs):
        title = str(args[2] if len(args) > 2 else kwargs.get("title", ""))
        return QColor("#224466" if "Experimental" in title else "#228833")

    monkeypatch.setattr(
        "saxshell.saxs.ui.project_setup_tab.QColorDialog.getColor",
        _pick_color,
    )

    tab._choose_data_trace_color("experimental")
    tab._choose_data_trace_color("solvent")
    tab.solvent_trace_visible_checkbox.setChecked(False)

    preview_axis = tab.component_figure.axes[0]
    labels = [line.get_label() for line in preview_axis.get_lines()]
    experimental_line = next(
        line
        for line in preview_axis.get_lines()
        if line.get_label() == "Experimental data"
    )

    assert tab.experimental_trace_visible_checkbox.isChecked()
    assert not tab.solvent_trace_visible_checkbox.isChecked()
    assert to_hex(experimental_line.get_color()) == "#224466"
    assert "Solvent data" not in labels

    window.save_project_state()

    reloaded_settings = SAXSProjectManager().load_project(project_dir)
    assert reloaded_settings.experimental_trace_color == "#224466"
    assert reloaded_settings.solvent_trace_color == "#228833"
    assert reloaded_settings.experimental_trace_visible is True
    assert reloaded_settings.solvent_trace_visible is False


def test_project_setup_component_overlay_uses_secondary_y_axis(qapp, tmp_path):
    del qapp
    q_values = np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float)
    data_path = tmp_path / "exp_overlay.txt"
    np.savetxt(
        data_path,
        np.column_stack(
            [
                q_values,
                np.asarray([100.0, 80.0, 55.0, 30.0], dtype=float),
            ]
        ),
    )
    component_path = tmp_path / "A_no_motif.txt"
    _write_component_file(
        component_path,
        q_values,
        np.asarray([4.8, 4.6, 4.4, 4.2], dtype=float),
    )

    summary = load_experimental_data_file(data_path)
    tab = ProjectSetupTab()
    tab.component_log_y_checkbox.setChecked(False)
    tab._apply_experimental_file(data_path, summary)
    tab.draw_component_plot([component_path])

    assert len(tab.component_figure.axes) == 2
    experimental_axis, component_axis = tab.component_figure.axes
    assert (
        experimental_axis.get_title()
        == "Experimental Data and SAXS Components"
    )
    assert experimental_axis.get_xlabel() == "q (Å⁻¹)"
    assert (
        experimental_axis.get_ylabel() == "Experimental Intensity (arb. units)"
    )
    assert component_axis.get_ylabel() == "Model Intensity (arb. units)"
    assert experimental_axis.get_legend() is not None
    assert (
        component_axis.get_ylim()[1] - component_axis.get_ylim()[0] > 4.8 - 4.2
    )


def test_component_legend_toggle_hides_and_shows_legend(qapp, tmp_path):
    del qapp
    q_values = np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float)
    data_path = tmp_path / "exp_overlay_toggle.txt"
    np.savetxt(
        data_path,
        np.column_stack(
            [
                q_values,
                np.asarray([100.0, 80.0, 55.0, 30.0], dtype=float),
            ]
        ),
    )
    component_path = tmp_path / "PbI2_edge.txt"
    _write_component_file(
        component_path,
        q_values,
        np.asarray([4.8, 4.6, 4.4, 4.2], dtype=float),
    )

    summary = load_experimental_data_file(data_path)
    tab = ProjectSetupTab()
    tab._apply_experimental_file(data_path, summary)
    tab.draw_component_plot([component_path])

    assert tab.component_figure.axes[0].get_legend() is not None

    tab.component_legend_toggle_button.setChecked(False)

    assert tab.component_figure.axes[0].get_legend() is None


def test_component_autoscale_to_model_range_uses_model_q_limits(
    qapp, tmp_path
):
    del qapp
    experimental_q = np.asarray([0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
    data_path = tmp_path / "exp_overlay_autoscale.txt"
    np.savetxt(
        data_path,
        np.column_stack(
            [
                experimental_q,
                np.asarray([150.0, 120.0, 95.0, 70.0, 40.0, 20.0]),
            ]
        ),
    )
    component_path = tmp_path / "PbI2_edge.txt"
    model_q = np.asarray([0.08, 0.12, 0.18], dtype=float)
    _write_component_file(
        component_path,
        model_q,
        np.asarray([4.8, 4.6, 4.2], dtype=float),
    )

    summary = load_experimental_data_file(data_path)
    tab = ProjectSetupTab()
    tab.component_log_y_checkbox.setChecked(False)
    tab._apply_experimental_file(data_path, summary)
    tab.draw_component_plot([component_path])

    tab.component_model_range_button.setChecked(True)

    x_limits = tab.component_figure.axes[0].get_xlim()
    assert x_limits[0] == pytest.approx(0.08)
    assert x_limits[1] == pytest.approx(0.18)


def test_component_table_toggle_and_legend_pick_stay_synced(qapp, tmp_path):
    del qapp
    q_values = np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float)
    data_path = tmp_path / "exp_overlay_sync.txt"
    np.savetxt(
        data_path,
        np.column_stack(
            [
                q_values,
                np.asarray([100.0, 80.0, 55.0, 30.0], dtype=float),
            ]
        ),
    )
    component_path = tmp_path / "PbI2_edge.txt"
    _write_component_file(
        component_path,
        q_values,
        np.asarray([4.8, 4.6, 4.4, 4.2], dtype=float),
    )

    summary = load_experimental_data_file(data_path)
    tab = ProjectSetupTab()
    tab._apply_experimental_file(data_path, summary)
    tab.apply_cluster_import_data(
        ["Pb", "I"],
        [
            {
                "structure": "PbI2",
                "motif": "edge",
                "count": 7,
                "weight": 0.1239,
                "atom_fraction_percent": 44.44,
                "structure_fraction_percent": 12.39,
            }
        ],
    )
    tab.draw_component_plot([component_path])

    visibility_item = tab.recognized_clusters_table.item(0, 6)
    color_item = tab.recognized_clusters_table.item(0, 7)

    assert visibility_item.checkState() == Qt.CheckState.Checked
    assert color_item.text() != "--"

    visibility_item.setCheckState(Qt.CheckState.Unchecked)

    assert not tab._component_line_lookup["PbI2_edge"].get_visible()
    assert tab._component_legend_lookup[
        "PbI2_edge"
    ].get_alpha() == pytest.approx(0.25)

    event = type(
        "LegendEvent",
        (),
        {"artist": tab._component_legend_lookup["PbI2_edge"]},
    )()
    tab._handle_component_legend_pick(event)

    refreshed_item = tab.recognized_clusters_table.item(0, 6)
    assert tab._component_line_lookup["PbI2_edge"].get_visible()
    assert refreshed_item.checkState() == Qt.CheckState.Checked


def test_prefit_plot_shows_solvent_contribution_and_legend_pick_toggles_model(
    qapp, tmp_path
):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    solvent_q = np.linspace(0.05, 0.3, 8)
    solvent_intensity = np.linspace(1.5, 2.2, 8)
    solvent_path = tmp_path / "prefit_solvent_trace.dat"
    np.savetxt(solvent_path, np.column_stack([solvent_q, solvent_intensity]))

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.solvent_data_path = str(solvent_path)
    settings.copied_solvent_data_file = None
    manager.save_project(settings)

    window = SAXSMainWindow(initial_project_dir=project_dir)
    entries = window.prefit_workflow.load_parameter_entries()
    for entry in entries:
        if entry.name == "solv_w":
            entry.value = 0.5
        if entry.name == "scale":
            entry.value = 2e-3

    evaluation = window.prefit_workflow.evaluate(entries)
    window.prefit_tab.plot_evaluation(evaluation)

    top_axis = window.prefit_tab.figure.axes[0]
    labels = [line.get_label() for line in top_axis.get_lines()]
    model_line = next(
        line for line in top_axis.get_lines() if line.get_label() == "Model"
    )

    assert "Solvent contribution" in labels

    event = type(
        "LegendEvent",
        (),
        {"artist": window.prefit_tab._legend_handle_lookup["Model"]},
    )()
    window.prefit_tab._handle_legend_pick(event)

    assert not model_line.get_visible()
    assert window.prefit_tab._legend_handle_lookup[
        "Model"
    ].get_alpha() == pytest.approx(0.25)


def test_experimental_status_note_is_wrapped_and_multiline(qapp):
    del qapp
    tab = ProjectSetupTab()

    assert tab.data_status_label.wordWrap()
    assert tab.data_status_label.minimumHeight() >= 72
    assert "\n" in tab.data_status_label.text()


def test_project_setup_uses_scrollable_resizable_side_panes(qapp):
    del qapp
    tab = ProjectSetupTab()

    assert tab._pane_splitter.count() == 2
    assert tab._pane_splitter.widget(0) is tab._left_scroll_area
    assert tab._pane_splitter.widget(1) is tab._right_scroll_area
    assert tab._left_scroll_area.widget() is not None
    assert tab._right_scroll_area.widget() is not None
    assert (
        tab._right_scroll_area.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        tab.component_group.parentWidget() is tab._right_scroll_area.widget()
    )
    assert tab.prior_group.parentWidget() is tab._right_scroll_area.widget()


def test_project_setup_activity_progress_updates(qapp):
    del qapp
    tab = ProjectSetupTab()

    tab.start_activity_progress(8, "Importing cluster files...")
    assert tab.activity_progress_label.text() == "Importing cluster files..."
    assert tab.activity_progress_bar.maximum() == 8
    assert tab.activity_progress_bar.value() == 0

    tab.update_activity_progress(3, 8, "Building SAXS components...")
    assert tab.activity_progress_label.text() == "Building SAXS components..."
    assert tab.activity_progress_bar.value() == 3

    tab.finish_activity_progress("Prior-weight generation complete.")
    assert (
        tab.activity_progress_label.text()
        == "Prior-weight generation complete."
    )
    assert tab.activity_progress_bar.value() == 8

    tab.reset_activity_progress()
    assert tab.activity_progress_label.text() == "Progress: idle"
    assert tab.activity_progress_bar.maximum() == 1
    assert tab.activity_progress_bar.value() == 0


def test_recognized_clusters_weight_is_truncated_to_three_decimals(qapp):
    del qapp
    tab = ProjectSetupTab()
    tab._populate_recognized_clusters_table(
        [
            {
                "structure": "Pb2I4",
                "motif": "edge",
                "count": 7,
                "weight": 0.1239,
                "atom_fraction_percent": 44.44,
                "structure_fraction_percent": 12.39,
            }
        ]
    )

    assert tab.recognized_clusters_table.item(0, 3).text() == "0.123"


def test_project_setup_tracks_selected_experimental_columns(qapp, tmp_path):
    del qapp
    data_path = tmp_path / "exp_project_columns.txt"
    data_path.write_text(
        "intensity_trace\tq_trace\terror_trace\n"
        "10.0\t0.05\t0.1\n"
        "9.0\t0.10\t0.2\n",
        encoding="utf-8",
    )

    summary = load_experimental_data_file(
        data_path,
        q_column=1,
        intensity_column=0,
        error_column=2,
    )
    tab = ProjectSetupTab()
    tab._apply_experimental_file(data_path, summary)

    assert tab.experimental_q_column() == 1
    assert tab.experimental_intensity_column() == 0
    assert tab.experimental_error_column() == 2
    assert "q=q_trace" in tab.data_status_label.text()
    assert "intensity=intensity_trace" in tab.data_status_label.text()
    assert "error=error_trace" in tab.data_status_label.text()


def test_prefit_plot_log_axes_toggle_independently(qapp, tmp_path):
    del qapp
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    window = SAXSMainWindow(initial_project_dir=project_dir)

    top_axis, bottom_axis = window.prefit_tab.figure.axes
    assert top_axis.get_xscale() == "log"
    assert top_axis.get_yscale() == "log"
    assert bottom_axis.get_xscale() == "log"

    window.prefit_tab.log_x_checkbox.setChecked(False)
    window.prefit_tab.log_y_checkbox.setChecked(False)

    top_axis, bottom_axis = window.prefit_tab.figure.axes
    assert top_axis.get_xscale() == "linear"
    assert top_axis.get_yscale() == "linear"
    assert bottom_axis.get_xscale() == "linear"


def test_use_experimental_grid_crops_to_nearest_available_q_values(tmp_path):
    manager = SAXSProjectManager()
    settings = ProjectSettings(
        project_name="demo_project",
        project_dir=str(tmp_path / "demo_project"),
        use_experimental_grid=True,
        q_min=0.081,
        q_max=0.171,
    )
    summary = ExperimentalDataSummary(
        path=tmp_path / "exp_demo.txt",
        q_values=np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float),
        intensities=np.asarray([10.0, 8.0, 6.0, 4.0], dtype=float),
    )

    q_grid = manager._build_q_grid(settings, summary)

    assert np.allclose(q_grid, [0.08, 0.12, 0.18])


def test_scan_cluster_inventory_reports_progress_and_rows(tmp_path):
    cluster_dir = tmp_path / "clusters"
    structure_dir = cluster_dir / "Pb2I4"
    structure_dir.mkdir(parents=True)
    (structure_dir / "frame_0001.xyz").write_text(
        "2\ncomment\nPb 0.0 0.0 0.0\nI 1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    manager = SAXSProjectManager()
    events: list[tuple[int, int, str]] = []

    result = manager.scan_cluster_inventory(
        cluster_dir,
        progress_callback=lambda processed, total, message: events.append(
            (processed, total, message)
        ),
    )

    assert isinstance(result, ClusterImportResult)
    assert result.available_elements == ["I", "Pb"]
    assert result.total_files == 1
    assert result.cluster_rows[0]["structure"] == "Pb2I4"
    assert events[0][2] == "Importing cluster files..."
    assert events[-1][2] == "Cluster import complete."


def test_cluster_inventory_cache_reuses_previous_scan(tmp_path, monkeypatch):
    cluster_dir = tmp_path / "clusters"
    structure_dir = cluster_dir / "Pb2I4"
    structure_dir.mkdir(parents=True)
    (structure_dir / "frame_0001.xyz").write_text(
        "3\ncomment\nPb 0.0 0.0 0.0\nI 1.0 0.0 0.0\nO 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    manager = SAXSProjectManager()

    initial = manager.scan_cluster_inventory(cluster_dir)

    def _fail_if_rescanned(_path):
        raise AssertionError("cluster inventory should have reused its cache")

    monkeypatch.setattr(
        project_module,
        "scan_structure_elements",
        _fail_if_rescanned,
    )

    cached = manager._collect_cluster_inventory(cluster_dir)

    assert cached.available_elements == initial.available_elements
    assert cached.total_files == initial.total_files
