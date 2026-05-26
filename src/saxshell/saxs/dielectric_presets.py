from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DielectricConstantPreset:
    key: str
    display_name: str
    solvent_name: str
    value: float
    source_label: str
    source_url: str

    @property
    def combo_label(self) -> str:
        return f"{self.display_name} ({self.value})"


LIBRETEXTS_SOLVENT_PROPERTIES_URL = (
    "https://chem.libretexts.org/Ancillary_Materials/Reference/"
    "Reference_Tables/Solvents/S1%3A_Solvent_Properties"
)

DIELECTRIC_CONSTANT_PRESETS: tuple[DielectricConstantPreset, ...] = (
    DielectricConstantPreset(
        key="DMF",
        display_name="DMF",
        solvent_name="N,N-dimethylformamide",
        value=36.7,
        source_label="Chemistry LibreTexts S1 solvent properties table",
        source_url=LIBRETEXTS_SOLVENT_PROPERTIES_URL,
    ),
    DielectricConstantPreset(
        key="DMSO",
        display_name="DMSO",
        solvent_name="dimethyl sulfoxide",
        value=47.0,
        source_label="Chemistry LibreTexts S1 solvent properties table",
        source_url=LIBRETEXTS_SOLVENT_PROPERTIES_URL,
    ),
    DielectricConstantPreset(
        key="NMP",
        display_name="NMP",
        solvent_name="N-methyl-2-pyrrolidone",
        value=32.0,
        source_label="Chemistry LibreTexts S1 solvent properties table",
        source_url=LIBRETEXTS_SOLVENT_PROPERTIES_URL,
    ),
    DielectricConstantPreset(
        key="ACN",
        display_name="ACN",
        solvent_name="acetonitrile",
        value=37.5,
        source_label="Chemistry LibreTexts S1 solvent properties table",
        source_url=LIBRETEXTS_SOLVENT_PROPERTIES_URL,
    ),
    DielectricConstantPreset(
        key="water",
        display_name="Water",
        solvent_name="water",
        value=78.54,
        source_label="Chemistry LibreTexts S1 solvent properties table",
        source_url=LIBRETEXTS_SOLVENT_PROPERTIES_URL,
    ),
)

DIELECTRIC_CONSTANT_PRESETS_BY_KEY = {
    preset.key: preset for preset in DIELECTRIC_CONSTANT_PRESETS
}
