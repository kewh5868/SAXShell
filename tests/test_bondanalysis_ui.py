import os
import shutil

import pytest
from matplotlib.colors import to_hex
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QScrollArea,
    QSplitter,
)

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisWorkflow,
    BondPairDefinition,
)
from saxshell.bondanalysis.results import build_plot_request, load_result_index
from saxshell.bondanalysis.ui.main_window import BondAnalysisMainWindow
from saxshell.bondanalysis.ui.plot_window import BondAnalysisPlotWindow
from saxshell.saxs.project_manager import SAXSProjectManager


def _write_xyz_cluster(path, atoms):
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n")


def _build_bondanalysis_output(tmp_path):
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi3_dir = clusters_dir / "PbI3"
    pbi2_dir.mkdir(parents=True)
    pbi3_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbi3_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
            ("I", 0.0, 0.0, 2.0),
        ],
    )

    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 3.1)],
        angle_triplets=[AngleTripletDefinition("Pb", "I", "I", 3.1, 3.1)],
        output_dir=tmp_path / "bondanalysis_results",
    )
    result = workflow.run()
    return clusters_dir, result.output_dir


def _find_results_leaf(window, category_label, distribution_label, leaf_label):
    for top_index in range(window.results_tree.topLevelItemCount()):
        category_item = window.results_tree.topLevelItem(top_index)
        if category_item.text(0) != category_label:
            continue
        for group_index in range(category_item.childCount()):
            distribution_item = category_item.child(group_index)
            if distribution_item.text(0) != distribution_label:
                continue
            for leaf_index in range(distribution_item.childCount()):
                leaf_item = distribution_item.child(leaf_index)
                if leaf_item.text(0) == leaf_label:
                    return leaf_item
    raise AssertionError(
        f"Unable to find results leaf {category_label}/{distribution_label}/{leaf_label}"
    )


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_bondanalysis_main_window_prefills_cluster_types_and_output_dir(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    assert window.cluster_type_list.count() == 2
    assert window.use_checked_cluster_types_box.isChecked()
    assert (
        window.cluster_type_list.item(0).checkState() == Qt.CheckState.Checked
    )
    assert (
        window.cluster_type_list.item(1).checkState() == Qt.CheckState.Checked
    )
    assert window.output_dir_edit.text().endswith(
        "bondanalysis_clusters_splitxyz0001"
    )
    preset_names = [
        window.preset_combo.itemText(index)
        for index in range(window.preset_combo.count())
    ]
    assert "DMSO (Built-in)" in preset_names
    assert "DMF (Built-in)" in preset_names
    assert "Analyzing checked cluster types: PbI2, PbO" in (
        window.selection_box.toPlainText()
    )
    preview_text = window.selection_box.toPlainText()
    assert "Cluster types detected: 2" in preview_text
    assert "Checked cluster types: 2" in preview_text
    assert "Displacement analysis: deprecated" in preview_text

    window.load_preset("DMSO")

    assert window.bond_pair_table.rowCount() == 7
    assert window.bond_pair_table.item(0, 0).text() == "Pb"
    assert window.bond_pair_table.item(0, 1).text() == "I"
    assert window.bond_pair_table.item(0, 2).text() == "4"
    assert window.angle_triplet_table.rowCount() == 5
    assert window.angle_triplet_table.item(2, 0).text() == "O"
    assert window.angle_triplet_table.item(2, 1).text() == "Pb"
    assert window.angle_triplet_table.item(2, 2).text() == "S"
    window.close()


def test_bondanalysis_main_window_shows_compact_project_status_and_registers_clusters_dir(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(
        initial_clusters_dir=clusters_dir,
        initial_project_dir=project_dir,
    )
    saved_settings = manager.load_project(project_dir)

    assert window.project_banner is None
    assert window.project_status_label is not None
    assert project_dir.name in window.project_status_label.toolTip()
    assert str(project_dir) in window.project_status_label.full_text()
    assert window.project_status_label.parent() is window.statusBar()
    assert saved_settings.resolved_clusters_dir == clusters_dir.resolve()
    window.close()


def test_bondanalysis_main_window_saves_custom_presets_across_sessions(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    preset_file = tmp_path / "bondanalysis_presets.json"
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(preset_file),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.load_preset("DMF")
    window.bond_pair_table.item(0, 2).setText("4.5")
    window.save_current_preset("DMF Custom")

    assert preset_file.exists()

    reloaded_window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    preset_names = [
        reloaded_window.preset_combo.itemText(index)
        for index in range(reloaded_window.preset_combo.count())
    ]
    assert "DMF Custom" in preset_names

    reloaded_window.load_preset("DMF Custom")

    assert reloaded_window.bond_pair_table.item(0, 2).text() == "4.5"


def test_bondanalysis_main_window_uses_checked_cluster_types_filter(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.use_checked_cluster_types_box.setChecked(True)
    window.cluster_type_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert window._selected_cluster_types() == ["PbI2"]
    preview_text = window.selection_box.toPlainText()
    assert "Checked cluster types: 1" in preview_text
    assert "Analyzing checked cluster types: PbI2" in preview_text


def test_bondanalysis_results_tree_groups_distributions_by_type(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    assert window.results_tree.topLevelItemCount() == 2
    all_bond_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "all")
    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    all_angle_leaf = _find_results_leaf(
        window,
        "Bond Angles",
        "I-Pb-I",
        "all",
    )

    assert all_bond_leaf.text(2) == "5"
    assert pb_i2_leaf.text(2) == "2"
    assert all_angle_leaf.text(2) == "4"


def test_bondanalysis_can_reload_existing_results_folder_without_recomputing(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    shutil.rmtree(clusters_dir)

    window = BondAnalysisMainWindow()
    window.load_existing_results_dir(output_dir)

    assert window.output_dir_edit.text() == str(output_dir)
    assert window.clusters_dir_edit.text() == str(clusters_dir)
    assert window.bond_pair_table.rowCount() == 1
    assert window.bond_pair_table.item(0, 0).text() == "Pb"
    assert window.bond_pair_table.item(0, 1).text() == "I"
    assert window.angle_triplet_table.rowCount() == 1
    assert window.angle_triplet_table.item(0, 0).text() == "Pb"
    assert window.results_tree.topLevelItemCount() == 2
    assert (
        "Loaded existing bondanalysis folder" in window.log_box.toPlainText()
    )


def test_bondanalysis_results_tree_selection_updates_ready_status(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    window.results_tree.clearSelection()
    pb_i2_leaf.setSelected(True)
    qapp.processEvents()

    assert (
        window.results_status_label.text()
        == "Ready to open Pb-I for PbI2 in a separate plot window."
    )


def test_bondanalysis_open_selected_window_can_overlay_selected_clusters_and_all_leaf(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    pb_i3_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI3")
    window.results_tree.clearSelection()
    pb_i2_leaf.setSelected(True)
    pb_i3_leaf.setSelected(True)
    qapp.processEvents()
    window._open_selected_plot_window()

    assert len(window._plot_windows) == 1
    overlay_window = window._plot_windows[0]
    legend = overlay_window.axis.get_legend()
    assert legend is not None
    assert {"PbI2", "PbI3", "Mean", "Median"} <= {
        text.get_text() for text in legend.get_texts()
    }
    assert overlay_window.series_color_container.isVisible()
    assert overlay_window.transparency_spin.isEnabled()

    all_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "all")
    window.results_tree.clearSelection()
    all_leaf.setSelected(True)
    qapp.processEvents()
    window._open_selected_plot_window()

    assert len(window._plot_windows) == 1
    all_window = window._plot_windows[0]
    assert all_window.tab_widget.count() == 2
    assert all_window.tab_widget.currentIndex() == 1
    assert len(all_window.plot_request.series) == 1
    assert all_window.plot_request.series[0].label == "all"
    assert all_window.plot_request.series[0].values.size == 5
    all_legend = all_window.axis.get_legend()
    assert all_legend is not None
    assert {"all", "Mean", "Median"} <= {
        text.get_text() for text in all_legend.get_texts()
    }
    assert len(all_window.axis.patches) > 0
    assert any("Mean:" in text.get_text() for text in all_window.axis.texts)
    assert not all_window.series_color_container.isVisible()


def test_bondanalysis_right_pane_is_scrollable_with_tree_above_run_log(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    splitter = window.centralWidget().layout().itemAt(0).widget()

    assert isinstance(splitter, QSplitter)
    assert isinstance(splitter.widget(1), QScrollArea)

    right_panel = splitter.widget(1).widget()
    right_layout = right_panel.layout()
    browser_log_panel = right_layout.itemAt(2).widget()
    browser_log_layout = browser_log_panel.layout()

    assert isinstance(right_layout.itemAt(0).widget(), QGroupBox)
    assert isinstance(right_layout.itemAt(1).widget(), QGroupBox)
    assert isinstance(browser_log_layout.itemAt(0).widget(), QGroupBox)
    assert isinstance(browser_log_layout.itemAt(1).widget(), QGroupBox)
    assert (
        browser_log_layout.itemAt(0).widget().title()
        == "Computed Distributions"
    )
    assert browser_log_layout.itemAt(1).widget().title() == "Run Log"


def test_bondanalysis_can_open_selected_plot_in_standalone_window(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    window.results_tree.clearSelection()
    pb_i2_leaf.setSelected(True)
    qapp.processEvents()
    window._open_selected_plot_window()

    assert len(window._plot_windows) == 1
    plot_window = window._plot_windows[0]
    assert isinstance(plot_window, BondAnalysisPlotWindow)
    assert plot_window.tab_widget.count() == 1
    assert "PbI2" in plot_window.windowTitle()
    assert plot_window.axis.get_title() == "PbI2 • Pb-I"


def test_bondanalysis_standalone_plot_window_saves_csv(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_request = build_plot_request(result_index, [group.all_leaf])
    plot_window = BondAnalysisPlotWindow(
        plot_request,
        default_output_dir=output_dir,
    )

    saved_csv_path = tmp_path / "saved_plot.csv"
    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.plot_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(saved_csv_path), "CSV Files (*.csv)"),
    )
    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.plot_window.QMessageBox.information",
        lambda *args, **kwargs: None,
    )

    plot_window.save_plot_data_as()

    assert saved_csv_path.exists()
    csv_lines = saved_csv_path.read_text().splitlines()
    assert csv_lines[0] == "Series,Value"
    assert all(line.startswith("all,") for line in csv_lines[1:])


def test_bondanalysis_plot_window_uses_horizontal_histogram_controls_and_stats(
    qapp,
    tmp_path,
    monkeypatch,
):
    del monkeypatch
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_request = build_plot_request(
        result_index,
        [group.cluster_leaves[0], group.cluster_leaves[1]],
    )
    plot_window = BondAnalysisPlotWindow(
        plot_request,
        default_output_dir=output_dir,
    )
    plot_window.show()
    qapp.processEvents()

    assert isinstance(plot_window.controls_widget.layout(), QHBoxLayout)
    assert not hasattr(plot_window, "plot_mode_combo")
    assert plot_window.bin_size_spin.isEnabled()
    assert plot_window.transparency_spin.isEnabled()
    assert plot_window.series_color_container.isVisible()

    plot_window.bin_size_spin.setValue(0.2)
    plot_window.transparency_spin.setValue(0.35)
    plot_window.refresh_plot()

    assert len(plot_window.axis.patches) > 0
    assert any(
        line.get_linestyle() == "--" and to_hex(line.get_color()) == "#000000"
        for line in plot_window.axis.lines
    )
    assert any(
        line.get_linestyle() == "--" and to_hex(line.get_color()) == "#ff0000"
        for line in plot_window.axis.lines
    )
    assert any(
        "Mean:" in text.get_text()
        and "Median:" in text.get_text()
        and "Mode:" in text.get_text()
        for text in plot_window.axis.texts
    )


def test_bondanalysis_plot_window_tabs_switch_with_arrow_keys(
    qapp,
    tmp_path,
    monkeypatch,
):
    del monkeypatch
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_window = BondAnalysisPlotWindow(
        build_plot_request(result_index, [group.cluster_leaves[0]]),
        default_output_dir=output_dir,
    )
    plot_window.add_plot_request(
        build_plot_request(result_index, [group.cluster_leaves[1]])
    )
    plot_window.show()
    plot_window.activateWindow()
    plot_window.setFocus()
    qapp.processEvents()

    assert plot_window.tab_widget.count() == 2
    assert plot_window.tab_widget.currentIndex() == 1

    QTest.keyClick(plot_window, Qt.Key.Key_Left)
    qapp.processEvents()
    assert plot_window.tab_widget.currentIndex() == 0
    assert plot_window.axis.get_title() == "PbI2 • Pb-I"

    QTest.keyClick(plot_window, Qt.Key.Key_Right)
    qapp.processEvents()
    assert plot_window.tab_widget.currentIndex() == 1
    assert plot_window.axis.get_title() == "PbI3 • Pb-I"


def test_bondanalysis_overlay_series_list_reorders_histogram_stacking(
    qapp,
    tmp_path,
    monkeypatch,
):
    del monkeypatch
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_window = BondAnalysisPlotWindow(
        build_plot_request(
            result_index,
            [group.cluster_leaves[0], group.cluster_leaves[1]],
        ),
        default_output_dir=output_dir,
    )
    plot_window.show()
    qapp.processEvents()

    assert [
        plot_window.series_color_list.item(i).text() for i in range(2)
    ] == [
        "PbI2",
        "PbI3",
    ]

    moved_item = plot_window.series_color_list.takeItem(0)
    plot_window.series_color_list.insertItem(1, moved_item)
    plot_window.current_plot_tab._on_series_order_changed()
    qapp.processEvents()

    assert [
        plot_window.series_color_list.item(i).text() for i in range(2)
    ] == [
        "PbI3",
        "PbI2",
    ]
    legend_labels = [
        text.get_text() for text in plot_window.axis.get_legend().get_texts()
    ]
    assert legend_labels[:2] == ["PbI3", "PbI2"]
