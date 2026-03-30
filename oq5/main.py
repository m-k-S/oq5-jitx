"""
OpenQuantum V5 Control Board — JITX Port
=========================================
Ported from atopile (.ato) to JITX (Python).

Three-stage power conditioning system for quantum experiment control:
  - Channel 1: 15V → 8.6V (buck) → 7V (LDO) — clean laser driver supply
  - Channel 2: 15V → 7V (buck) → 5V (LDO) — RedPitaya supply
  - Channel 3: 15V → 7V (buck) → 5V (LDO) — VCO supply

Plus: Koheron CTL200 laser driver, bias tee, DB15 interconnect, SMA RF input.
"""

from jitx import Design
from jitx.board import Board
from jitx.circuit import Circuit
from jitx.component import Component
from jitx.common import Power
from jitx.layerindex import Side
from jitx.net import Net, Port
from jitx.feature import Cutout
from jitx.landpattern import Landpattern, Pad, PadMapping
from jitx.shapes.composites import rectangle
from jitx.shapes.primitive import Circle, Polygon
from jitx.si import RoutingStructure, symmetric_routing_layers
from jitx.stackup import Conductor, Dielectric, Stackup
from jitx.substrate import FabricationConstraints, Substrate
from jitx.units import ohm
from jitx.via import Via, ViaType
from jitxlib.parts import Resistor, Capacitor, Inductor
from jitxlib.symbols.box import BoxSymbol


# =============================================================================
# Board, Stackup & Substrate (matching KiCad 2-layer 1.6 mm FR4 / ENIG)
# =============================================================================

class Soldermask(Dielectric):
    dielectric_coefficient = 3.3
    loss_tangent = 0.02


class FR4(Dielectric):
    dielectric_coefficient = 4.5
    loss_tangent = 0.02


class Copper1oz(Conductor):
    roughness = 0.001


copper1oz = Copper1oz(thickness=0.035)


class OQ5Stackup(Stackup):
    top_mask = Soldermask(thickness=0.01)
    top = copper1oz
    core = FR4(thickness=1.51)
    bottom = copper1oz
    bottom_mask = Soldermask(thickness=0.01)


class OQ5FabConstraints(FabricationConstraints):
    min_copper_width = 0.12
    min_copper_copper_space = 0.12
    min_copper_hole_space = 0.25
    min_copper_edge_space = 0.5
    min_annular_ring = 0.1
    min_drill_diameter = 0.3
    min_silkscreen_width = 0.127
    min_pitch_leaded = 0.35
    min_pitch_bga = 0.35
    max_board_width = 500.0
    max_board_height = 500.0
    min_silk_solder_mask_space = 0.15
    min_silkscreen_text_height = 1.0
    solder_mask_registration = 0.15
    min_soldermask_opening = 0.152
    min_soldermask_bridge = 0.102
    min_th_pad_expand_outer = 0.1
    min_hole_to_hole = 0.25
    min_pth_pin_solder_clearance = 3.0


class OQ5Substrate(Substrate):
    stackup = OQ5Stackup()
    constraints = OQ5FabConstraints()

    class THVia(Via):
        start_layer = 0
        stop_layer = -1
        diameter = 0.6
        hole_diameter = 0.3
        type = ViaType.MechanicalDrill

    RS_50 = RoutingStructure(
        impedance=50 * ohm,
        layers=symmetric_routing_layers({
            0: RoutingStructure.Layer(
                trace_width=0.12,
                clearance=0.2,
                velocity=191335235228,
                insertion_loss=0.0178,
            )
        }),
    )


class OQ5Board(Board):
    """200 x 150 mm board — adjust to match your actual outline."""
    shape = rectangle(200, 150, radius=3)


# =============================================================================
# Pad Helpers & Landpattern Definitions (from KiCad footprints)
# =============================================================================
# KiCad uses Y-down; JITX uses Y-up. All Y coords are negated below.

class _SMDRect(Pad):
    """Surface-mount rectangular pad."""
    def __init__(self, w, h):
        self.shape = rectangle(w, h)


class _SMDPoly(Pad):
    """Surface-mount pad with custom polygon copper shape."""
    def __init__(self, vertices):
        self.shape = Polygon(vertices)


class _THCircle(Pad):
    """Through-hole circular pad."""
    def __init__(self, d, drill):
        self.shape = Circle(diameter=d)
        self.cutout = Cutout(Circle(diameter=drill))


class _THRect(Pad):
    """Through-hole rectangular pad."""
    def __init__(self, w, h, drill):
        self.shape = rectangle(w, h)
        self.cutout = Cutout(Circle(diameter=drill))


# --- TI TPSM863252RDXR  (QFN-7 power module, 4.0 x 3.3 mm) ----------------

class TPSM863252Landpattern(Landpattern):
    # Pads 1-4: complex polygon copper (internal power planes)
    p1 = _SMDPoly([                          # VIN
        (0.1, 0.26), (0.1, 0.54), (-0.32, 0.54), (-0.47, 0.39),
        (-0.47, 0.3), (-0.87, 0.3), (-0.87, 0.08), (-0.47, 0.08),
        (-0.47, -0.2), (-0.87, -0.2), (-0.87, -0.42), (-0.47, -0.42),
        (-0.47, -0.54), (0.67, -0.54), (0.67, -0.12), (0.48, -0.12),
    ]).at(-0.97, 0.7)
    p2 = _SMDPoly([                          # SW
        (0.6, 1.09), (0.6, -1.1), (-0.4, -1.1), (-0.4, -0.87),
        (-0.8, -0.87), (-0.8, -0.66), (-0.4, -0.66), (-0.4, -0.37),
        (-0.8, -0.37), (-0.8, -0.16), (-0.4, -0.16), (-0.4, 0.13),
        (-0.8, 0.13), (-0.8, 0.34), (-0.4, 0.34), (-0.4, 0.63),
        (-0.8, 0.63), (-0.8, 0.84), (-0.4, 0.84), (-0.4, 1.09),
    ]).at(-1.05, -1.29)
    p3 = _SMDPoly([                          # VOUT
        (0.4, 0.87), (0.4, 1.1), (-0.6, 1.1), (-0.6, -1.09),
        (0.4, -1.09), (0.4, -0.84), (0.8, -0.84), (0.8, -0.63),
        (0.4, -0.63), (0.4, -0.34), (0.8, -0.34), (0.8, -0.13),
        (0.4, -0.13), (0.4, 0.16), (0.8, 0.16), (0.8, 0.37),
        (0.4, 0.37), (0.4, 0.66), (0.8, 0.66), (0.8, 0.87),
    ]).at(1.05, -1.29)
    p4 = _SMDPoly([                          # PGND
        (-0.11, 0.54), (-0.11, 0.26), (-0.49, -0.12), (-0.68, -0.12),
        (-0.68, -0.54), (0.46, -0.54), (0.46, -0.42), (0.86, -0.42),
        (0.86, -0.2), (0.46, -0.2), (0.46, 0.08), (0.86, 0.08),
        (0.86, 0.3), (0.46, 0.3), (0.46, 0.54),
    ]).at(0.98, 0.7)
    # Pads 5-7: small rectangular pads on bottom edge
    p5 = _SMDRect(0.22, 0.6).at(0.5, 1.29)    # PG
    p6 = _SMDRect(0.22, 0.6).at(0.0, 1.29)    # EN
    p7 = _SMDRect(0.22, 0.6).at(-0.5, 1.29)   # FB


# --- onsemi NCP5663DSADJR4G  (D2PAK-5, 10.0 x 9.2 mm) ---------------------

class NCP5663Landpattern(Landpattern):
    p1 = _SMDRect(3.5, 1.07).at(5.31, -3.4)   # Enable
    p2 = _SMDRect(3.5, 1.07).at(5.31, -1.7)   # VIN
    p3 = _SMDRect(3.5, 1.07).at(5.31, 0.0)    # GND
    p4 = _SMDRect(3.5, 1.07).at(5.31, 1.7)    # Vout
    p5 = _SMDRect(3.5, 1.07).at(5.31, 3.4)    # Adj
    p6 = _SMDRect(8.38, 10.66).at(-5.31, 0.0)  # Exposed/thermal pad


# --- Phoenix Contact 1717729  (2-pin, 7.62 mm pitch THT) -------------------

class PhoenixContact1717729LP(Landpattern):
    p1 = _THCircle(2.5, 1.6).at(-3.81, 0.0)
    p2 = _THCircle(2.5, 1.6).at(3.81, 0.0)


# --- Phoenix Contact 1729128  (2-pin, 5.08 mm pitch THT) -------------------

class PhoenixContact1729128LP(Landpattern):
    p1 = _THRect(2.2, 2.2, 1.5).at(-2.54, 0.0)
    p2 = _THCircle(2.2, 1.5).at(2.54, 0.0)


# --- Omron XM3B-1522-502  (DB15, 2.74 mm pitch THT) -----------------------

class DB15Landpattern(Landpattern):
    # Row 1 (pins 1-8)
    p1 = _THRect(1.57, 1.57, 1.1).at(9.59, 1.42)
    p2 = _THCircle(1.57, 1.1).at(6.85, 1.42)
    p3 = _THCircle(1.57, 1.1).at(4.11, 1.42)
    p4 = _THCircle(1.57, 1.1).at(1.37, 1.42)
    p5 = _THCircle(1.57, 1.1).at(-1.37, 1.42)
    p6 = _THCircle(1.57, 1.1).at(-4.11, 1.42)
    p7 = _THCircle(1.57, 1.1).at(-6.85, 1.42)
    p8 = _THCircle(1.57, 1.1).at(-9.59, 1.42)
    # Row 2 (pins 9-15)
    p9 = _THCircle(1.57, 1.1).at(8.22, -1.42)
    p10 = _THCircle(1.57, 1.1).at(5.48, -1.42)
    p11 = _THCircle(1.57, 1.1).at(2.74, -1.42)
    p12 = _THCircle(1.57, 1.1).at(0.0, -1.42)
    p13 = _THCircle(1.57, 1.1).at(-2.74, -1.42)
    p14 = _THCircle(1.57, 1.1).at(-5.48, -1.42)
    p15 = _THCircle(1.57, 1.1).at(-8.22, -1.42)
    # Mounting holes
    MH1 = _THCircle(5.0, 3.25).at(-16.66, 0.0)
    MH2 = _THCircle(5.0, 3.25).at(16.66, 0.0)


# --- Koheron CTL200  (20-pin THT connector) --------------------------------

class KoheronLandpattern(Landpattern):
    # All signal pads: 1.651 mm circle, 0.762 mm drill
    gnd1 = _THCircle(1.651, 0.762).at(-4.953, 2.032)
    ld_p = _THCircle(1.651, 0.762).at(-4.953, -0.508)
    ld_n = _THCircle(1.651, 0.762).at(-4.953, -3.048)
    gnd2 = _THCircle(1.651, 0.762).at(-4.953, -5.588)
    vl_p = _THCircle(1.651, 0.762).at(-4.953, -8.128)
    vl_n = _THCircle(1.651, 0.762).at(-4.953, -10.668)
    gnd3 = _THCircle(1.651, 0.762).at(-4.953, -13.208)
    pd_p = _THCircle(1.651, 0.762).at(-4.953, -15.748)
    pd_n = _THCircle(1.651, 0.762).at(-4.953, -18.288)
    gnd4 = _THCircle(1.651, 0.762).at(-4.953, -39.624)
    th_p = _THCircle(1.651, 0.762).at(-4.953, -42.164)
    th_n = _THCircle(1.651, 0.762).at(-4.953, -44.704)
    gnd5 = _THCircle(1.651, 0.762).at(-4.953, -47.244)
    tec_p1 = _THCircle(1.651, 0.762).at(-4.953, -49.784)
    tec_p2 = _THCircle(1.651, 0.762).at(-4.953, -52.324)
    tec_n1 = _THCircle(1.651, 0.762).at(-4.953, -54.864)
    tec_n2 = _THCircle(1.651, 0.762).at(-4.953, -57.404)
    gnd6 = _THCircle(1.651, 0.762).at(-4.953, -59.944)
    gnd7 = _THCircle(1.651, 0.762).at(38.49, -64.672)
    vs = _THCircle(1.651, 0.762).at(38.49, -67.212)


# --- RFENABLE  (2-pin THT header) ------------------------------------------

class RFEnableLandpattern(Landpattern):
    p1 = _THCircle(1.7, 1.0).at(0.0, 1.655)
    p2 = _THCircle(1.7, 1.0).at(0.0, -1.655)


# --- VTUNE  (4-pin THT header) ---------------------------------------------

class VTuneLandpattern(Landpattern):
    p1 = _THCircle(1.7, 1.0).at(0.0, 5.715)
    p2 = _THCircle(1.7, 1.0).at(0.0, 1.905)
    p3 = _THCircle(1.7, 1.0).at(0.0, -1.905)
    p4 = _THCircle(1.7, 1.0).at(0.0, -5.715)


# --- BAT Wireless SMA  (5-pin THT RF connector) ----------------------------

class SMALandpattern(Landpattern):
    sig = _THCircle(2.0, 1.35).at(0.0, 0.0)      # Center RF contact
    gnd1 = _THCircle(2.2, 1.5).at(-2.55, -2.55)   # Corner grounds
    gnd2 = _THCircle(2.2, 1.5).at(-2.55, 2.55)
    gnd3 = _THCircle(2.2, 1.5).at(2.55, 2.55)
    gnd4 = _THCircle(2.2, 1.5).at(2.55, -2.55)


# =============================================================================
# Custom Component Definitions (connectors and ICs without JITX library parts)
# =============================================================================

class TI_TPSM863252RDXR(Component):
    """TI TPSM863252 Synchronous Buck Module (QFN-7 package)."""
    mpn = "TPSM863252RDXR"
    manufacturer = "Texas Instruments"
    VIN = Port()
    SW = Port()
    VOUT = Port()
    FB = Port()
    EN = Port()
    PG = Port()
    PGND = Port()

    lp = TPSM863252Landpattern()
    symbol = BoxSymbol()

    def __init__(self):
        self.pad_mapping = PadMapping({
            self.VIN: self.lp.p1,
            self.SW: self.lp.p2,
            self.VOUT: self.lp.p3,
            self.PGND: self.lp.p4,
            self.PG: self.lp.p5,
            self.EN: self.lp.p6,
            self.FB: self.lp.p7,
        })


class NCP5663DSADJR4G(Component):
    """onsemi NCP5663 Adjustable LDO (D2PAK-5 package)."""
    mpn = "NCP5663DSADJR4G"
    manufacturer = "onsemi"
    VIN = Port()
    Vout = Port()
    Adj = Port()
    Enable = Port()
    GND = Port()
    EP = Port()  # Exposed pad

    lp = NCP5663Landpattern()
    symbol = BoxSymbol()

    def __init__(self):
        self.pad_mapping = PadMapping({
            self.Enable: self.lp.p1,
            self.VIN: self.lp.p2,
            self.GND: self.lp.p3,
            self.Vout: self.lp.p4,
            self.Adj: self.lp.p5,
            self.EP: self.lp.p6,
        })


class PhoenixContact1717729(Component):
    """Phoenix Contact 1717729 — 2-pin, 7.62 mm pitch terminal block."""
    mpn = "1717729"
    manufacturer = "Phoenix Contact"
    p1 = Port()
    p2 = Port()
    landpattern = PhoenixContact1717729LP()
    symbol = BoxSymbol()


class PhoenixContact1729128(Component):
    """Phoenix Contact 1729128 — 2-pin, 5.08 mm pitch terminal block."""
    mpn = "1729128"
    manufacturer = "Phoenix Contact"
    p1 = Port()
    p2 = Port()
    landpattern = PhoenixContact1729128LP()
    symbol = BoxSymbol()


class OmronXM3B1522502(Component):
    """Omron XM3B-1522-502 — DB15 connector."""
    mpn = "XM3B-1522-502"
    manufacturer = "Omron Electronics"
    p1 = Port()
    p2 = Port()
    p3 = Port()
    p4 = Port()
    p5 = Port()
    p6 = Port()
    p7 = Port()
    p8 = Port()
    p9 = Port()
    p10 = Port()
    p11 = Port()
    p12 = Port()
    p13 = Port()
    p14 = Port()
    p15 = Port()
    MH1 = Port()
    MH2 = Port()
    landpattern = DB15Landpattern()
    symbol = BoxSymbol()


class KoheronPackage(Component):
    """Koheron CTL200 laser controller connector."""
    mpn = "CTL200"
    manufacturer = "Koheron"
    vs = Port()
    gnd1 = Port()
    gnd2 = Port()
    gnd3 = Port()
    gnd4 = Port()
    gnd5 = Port()
    gnd6 = Port()
    gnd7 = Port()
    ld_p = Port()
    ld_n = Port()
    tec_p1 = Port()
    tec_p2 = Port()
    tec_n1 = Port()
    tec_n2 = Port()
    th_p = Port()
    th_n = Port()
    pd_p = Port()
    pd_n = Port()
    vl_p = Port()
    vl_n = Port()

    lp = KoheronLandpattern()
    symbol = BoxSymbol()

    def __init__(self):
        self.pad_mapping = PadMapping({
            self.vs: self.lp.vs,
            self.gnd1: self.lp.gnd1,
            self.gnd2: self.lp.gnd2,
            self.gnd3: self.lp.gnd3,
            self.gnd4: self.lp.gnd4,
            self.gnd5: self.lp.gnd5,
            self.gnd6: self.lp.gnd6,
            self.gnd7: self.lp.gnd7,
            self.ld_p: self.lp.ld_p,
            self.ld_n: self.lp.ld_n,
            self.tec_p1: self.lp.tec_p1,
            self.tec_p2: self.lp.tec_p2,
            self.tec_n1: self.lp.tec_n1,
            self.tec_n2: self.lp.tec_n2,
            self.th_p: self.lp.th_p,
            self.th_n: self.lp.th_n,
            self.pd_p: self.lp.pd_p,
            self.pd_n: self.lp.pd_n,
            self.vl_p: self.lp.vl_p,
            self.vl_n: self.lp.vl_n,
        })


class RFEnablePackage(Component):
    """RFENABLE daughterboard — 2-pin header."""
    p1 = Port()
    p2 = Port()
    landpattern = RFEnableLandpattern()
    symbol = BoxSymbol()


class VTunePackage(Component):
    """VTUNE daughterboard — 4-pin header."""
    p1 = Port()
    p2 = Port()
    p3 = Port()
    p4 = Port()
    landpattern = VTuneLandpattern()
    symbol = BoxSymbol()


class SMAConnector(Component):
    """BAT Wireless BWSMA-KE-Z001 — SMA RF connector."""
    mpn = "BWSMA-KE-Z001"
    manufacturer = "BAT Wireless"
    sig = Port()
    gnd1 = Port()
    gnd2 = Port()
    gnd3 = Port()
    gnd4 = Port()

    lp = SMALandpattern()
    symbol = BoxSymbol()

    def __init__(self):
        self.pad_mapping = PadMapping({
            self.sig: self.lp.sig,
            self.gnd1: self.lp.gnd1,
            self.gnd2: self.lp.gnd2,
            self.gnd3: self.lp.gnd3,
            self.gnd4: self.lp.gnd4,
        })


# =============================================================================
# Sub-Circuits
# =============================================================================

class LCPiFilter(Circuit):
    """
    LC Pi filter for power rail filtering.
    Topology: C1 → L → C2, with HF bypass caps on each side.
    Cutoff ~16 kHz (10 µH + 10 µF), −40 dB/decade rolloff.
    """
    power_in = Power()
    power_out = Power()

    def __init__(self):
        # --- Inductor ---
        self.inductor = Inductor(inductance=10e-6)  # 10 µH

        # --- Input caps (C1) ---
        self.c1_bulk = Capacitor(capacitance=10e-6)    # 10 µF bulk
        self.c1_hf = Capacitor(capacitance=100e-9)     # 100 nF HF bypass

        # --- Output caps (C2) ---
        self.c2_bulk = Capacitor(capacitance=10e-6)    # 10 µF bulk
        self.c2_hf = Capacitor(capacitance=100e-9)     # 100 nF HF bypass

        # --- Placement: C1 → L → C2 (left to right) ---
        self.c1_bulk.at(-6, 2)
        self.c1_hf.at(-6, -2)
        self.inductor.at(0, 0)
        self.c2_bulk.at(6, 2)
        self.c2_hf.at(6, -2)

        # --- Wiring ---
        # Input side caps: power_in.Vp ── C ── power_in.Vn
        self.nets = [
            self.power_in.Vp + self.c1_bulk.p1,
            self.c1_bulk.p2 + self.power_in.Vn,
            self.power_in.Vp + self.c1_hf.p1,
            self.c1_hf.p2 + self.power_in.Vn,
            # Series inductor: power_in.Vp ── L ── power_out.Vp
            self.power_in.Vp + self.inductor.p1,
            self.inductor.p2 + self.power_out.Vp,
            # Output side caps: power_out.Vp ── C ── power_out.Vn
            self.power_out.Vp + self.c2_bulk.p1,
            self.c2_bulk.p2 + self.power_out.Vn,
            self.power_out.Vp + self.c2_hf.p1,
            self.c2_hf.p2 + self.power_out.Vn,
            # Ground continuity
            self.power_in.Vn + self.power_out.Vn,
        ]


class BiasTee(Circuit):
    """
    Bias tee for combining DC and AC/RF signals.
    DC path: 470 µH inductor (passes DC, blocks AC).
    AC path: 100 nF capacitor (passes AC, blocks DC).
    Lower cutoff ~32 kHz, upper cutoff >10 MHz.
    """
    dc_in = Port()
    rf_in = Port()
    output = Port()

    def __init__(self):
        # DC path: inductor (RF choke)
        self.inductor = Inductor(inductance=470e-6)  # 470 µH
        # AC path: coupling capacitor
        self.cap = Capacitor(capacitance=100e-9)  # 100 nF

        # --- Placement ---
        self.inductor.at(-4, 0)
        self.cap.at(4, 0)

        self.nets = [
            # DC: dc_in ── L ── output
            self.dc_in + self.inductor.p1,
            self.inductor.p2 + self.output,
            # RF: rf_in ── C ── output
            self.rf_in + self.cap.p1,
            self.cap.p2 + self.output,
        ]


class TPSM863252(Circuit):
    """
    TI TPSM863252 Synchronous Buck Module sub-circuit.
    3 V to 17 V input, 0.6 V to 10 V output, up to 3 A.
    Parameterised by r_top_ohms to set the output voltage via the
    feedback divider: Vout = 0.6 V × (1 + R_top / R_bottom).
    """
    power_in = Power()
    power_out = Power()
    enable = Port()
    power_good = Port()

    def __init__(self, r_top_ohms: float):
        self.ic = TI_TPSM863252RDXR()

        # --- Input decoupling: 2 × 10 µF ---
        self.input_caps = [Capacitor(capacitance=10e-6) for _ in range(2)]
        # --- Output decoupling: 2 × 22 µF ---
        self.output_caps = [Capacitor(capacitance=22e-6) for _ in range(2)]

        # --- Feedback divider ---
        self.r_top = Resistor(resistance=r_top_ohms)
        self.r_bottom = Resistor(resistance=10e3)  # 10 kΩ

        # --- Enable pullup: 10 kΩ ---
        self.enable_pullup = Resistor(resistance=10e3)

        # --- Placement (IC centre, caps flanking, divider below) ---
        self.ic.at(0, 0)
        self.input_caps[0].at(-6, 3)
        self.input_caps[1].at(-6, -3)
        self.output_caps[0].at(6, 3)
        self.output_caps[1].at(6, -3)
        self.r_top.at(3, -7)
        self.r_bottom.at(3, -10)
        self.enable_pullup.at(-3, -7)

        # --- Nets ---
        self.nets = []

        # Power connections
        self.nets += [
            self.power_in.Vp + self.ic.VIN,
            self.power_in.Vn + self.ic.PGND,
            self.power_out.Vp + self.ic.VOUT,
            self.power_out.Vn + self.ic.PGND,
        ]

        # Input decoupling
        for cap in self.input_caps:
            self.nets += [
                self.power_in.Vp + cap.p1,
                cap.p2 + self.power_in.Vn,
            ]

        # Output decoupling
        for cap in self.output_caps:
            self.nets += [
                self.power_out.Vp + cap.p1,
                cap.p2 + self.power_out.Vn,
            ]

        # Feedback divider: VOUT → R_top → FB → R_bottom → GND
        self.nets += [
            self.power_out.Vp + self.r_top.p1,
            self.r_top.p2 + self.ic.FB,
            self.ic.FB + self.r_bottom.p1,
            self.r_bottom.p2 + self.power_out.Vn,
        ]

        # Enable pullup: VIN → 10 kΩ → EN
        self.nets += [
            self.power_in.Vp + self.enable_pullup.p1,
            self.enable_pullup.p2 + self.ic.EN,
        ]

        # External enable & power-good signals
        self.nets += [
            self.enable + self.ic.EN,
            self.power_good + self.ic.PG,
        ]

        # SW is an internal switching node — not routed externally
        self.ic.SW.no_connect()


class TPSM863252_8V6(TPSM863252):
    """15 V → 8.6 V buck. R_top = 133 kΩ, R_bottom = 10 kΩ.
    Vout = 0.6 × (1 + 133 k / 10 k) = 8.58 V."""
    def __init__(self):
        super().__init__(r_top_ohms=133e3)


class TPSM863252_7V(TPSM863252):
    """15 V → 7 V buck. R_top = 107 kΩ, R_bottom = 10 kΩ.
    Vout = 0.6 × (1 + 107 k / 10 k) = 7.02 V."""
    def __init__(self):
        super().__init__(r_top_ohms=107e3)


class NCP5663(Circuit):
    """
    onsemi NCP5663 Adjustable LDO sub-circuit.
    Up to 20 V input, 1.25 V reference, up to 3 A output.
    Parameterised by r_top_ohms: Vout = 1.25 V × (1 + R_top / R_bottom).
    """
    power_in = Power()
    power_out = Power()
    enable = Port()

    def __init__(self, r_top_ohms: float):
        self.ic = NCP5663DSADJR4G()

        # --- Input decoupling: 10 µF ---
        self.input_cap = Capacitor(capacitance=10e-6)
        # --- Output decoupling: 2 × 22 µF ---
        self.output_caps = [Capacitor(capacitance=22e-6) for _ in range(2)]

        # --- Feedback divider ---
        self.r_top = Resistor(resistance=r_top_ohms)
        self.r_bottom = Resistor(resistance=10e3)  # 10 kΩ

        # --- Enable pullup: 10 kΩ (default-on) ---
        self.enable_pullup = Resistor(resistance=10e3)

        # --- Placement (IC centre, caps flanking, divider below) ---
        self.ic.at(0, 0)
        self.input_cap.at(-8, 0)
        self.output_caps[0].at(8, 3)
        self.output_caps[1].at(8, -3)
        self.r_top.at(5, -7)
        self.r_bottom.at(5, -10)
        self.enable_pullup.at(-5, -7)

        # --- Nets ---
        self.nets = []

        # Power connections
        self.nets += [
            self.power_in.Vp + self.ic.VIN,
            self.power_in.Vn + self.ic.GND,
            self.power_out.Vp + self.ic.Vout,
            self.power_out.Vn + self.ic.GND,
            # Exposed pad to GND (thermal)
            self.ic.EP + self.ic.GND,
        ]

        # Input decoupling
        self.nets += [
            self.power_in.Vp + self.input_cap.p1,
            self.input_cap.p2 + self.power_in.Vn,
        ]

        # Output decoupling
        for cap in self.output_caps:
            self.nets += [
                self.power_out.Vp + cap.p1,
                cap.p2 + self.power_out.Vn,
            ]

        # Feedback divider: VOUT → R_top → Adj → R_bottom → GND
        self.nets += [
            self.power_out.Vp + self.r_top.p1,
            self.r_top.p2 + self.ic.Adj,
            self.ic.Adj + self.r_bottom.p1,
            self.r_bottom.p2 + self.power_out.Vn,
        ]

        # Enable pullup: VIN → 10 kΩ → Enable
        self.nets += [
            self.power_in.Vp + self.enable_pullup.p1,
            self.enable_pullup.p2 + self.ic.Enable,
        ]

        # External enable
        self.nets.append(self.enable + self.ic.Enable)


class NCP5663_7V(NCP5663):
    """8.6 V → 7 V LDO. R_top = 46 kΩ, R_bottom = 10 kΩ.
    Vout = 1.25 × (1 + 46 k / 10 k) = 7.0 V."""
    def __init__(self):
        super().__init__(r_top_ohms=46e3)


class NCP5663_5V(NCP5663):
    """7 V → 5 V LDO. R_top = 30 kΩ, R_bottom = 10 kΩ.
    Vout = 1.25 × (1 + 30 k / 10 k) = 5.0 V."""
    def __init__(self):
        super().__init__(r_top_ohms=30e3)


class KoheronLaserDriver(Circuit):
    """
    Koheron CTL200 Digital Laser Controller.
    Connections for laser diode, TEC, thermistor, and power.
    PD and VL pins left unconnected.
    """
    power = Power()
    ld_p = Port()
    ld_n = Port()
    tec_p = Port()
    tec_n = Port()
    th_p = Port()
    th_n = Port()

    def __init__(self):
        self.ic = KoheronPackage()
        self.ic.at(0, 0)

        # GND connections (7 ground pins)
        self.GND = (
            self.power.Vn
            + self.ic.gnd1 + self.ic.gnd2 + self.ic.gnd3 + self.ic.gnd4
            + self.ic.gnd5 + self.ic.gnd6 + self.ic.gnd7
        )

        self.nets = [
            # Power supply
            self.power.Vp + self.ic.vs,
            # Laser diode
            self.ld_p + self.ic.ld_p,
            self.ld_n + self.ic.ld_n,
            # TEC / Peltier (parallel pairs)
            self.tec_p + self.ic.tec_p1,
            self.tec_p + self.ic.tec_p2,
            self.tec_n + self.ic.tec_n1,
            self.tec_n + self.ic.tec_n2,
            # Thermistor
            self.th_p + self.ic.th_p,
            self.th_n + self.ic.th_n,
        ]

        # PD+, PD-, VL+, VL- left unconnected
        self.ic.pd_p.no_connect()
        self.ic.pd_n.no_connect()
        self.ic.vl_p.no_connect()
        self.ic.vl_n.no_connect()


# =============================================================================
# Top-Level Design
# =============================================================================

class OpenQuantumV5(Circuit):
    """
    OpenQuantum V5 Control Board — top-level circuit.

    Input : 15 V via Phoenix Contact terminal block, LC-pi filtered.
    Output: Three independent regulated rails + laser driver + RF path.
    """

    def __init__(self):
        # =====================================================================
        # Power rails (declared as Power bundles with .Vp / .Vn)
        # =====================================================================
        self.power_15v = Power()

        # === Channel 1: 15 V → 8.6 V → 7 V (clean laser driver supply) ===
        self.buck_7v = TPSM863252_8V6()
        self.power_8v6 = Power()
        self.ldo_7v = NCP5663_7V()
        self.power_7v = Power()

        # === Channel 2: 15 V → 7 V → 5 V (RedPitaya) ===
        self.buck_redpitaya = TPSM863252_7V()
        self.power_7v_redpitaya = Power()
        self.ldo_redpitaya = NCP5663_5V()
        self.power_5v_redpitaya = Power()

        # === Channel 3: 15 V → 7 V → 5 V (VCO) ===
        self.buck_vco = TPSM863252_7V()
        self.power_7v_vco = Power()
        self.ldo_vco = NCP5663_5V()
        self.power_5v_vco = Power()

        # === Koheron CTL200 Laser Driver ===
        self.koheron = KoheronLaserDriver()

        # === Filters ===
        self.filter_15v = LCPiFilter()        # Input 15 V filter
        self.filter_5v_rp = LCPiFilter()      # RedPitaya 5 V output filter

        # === Bias Tee ===
        self.bias_tee = BiasTee()

        # === Connectors ===
        self.j_15v_in = PhoenixContact1717729()        # 15 V input
        self.j_dps5005_1 = PhoenixContact1729128()     # DPS5005 tap 1
        self.j_dps5005_2 = PhoenixContact1729128()     # DPS5005 tap 2
        self.j_5v_redpitaya = PhoenixContact1729128()  # 5 V RedPitaya output
        self.db15 = OmronXM3B1522502()                 # DB15 connector
        self.rfenable = RFEnablePackage()               # RF enable daughterboard
        self.vtune_db = VTunePackage()                  # VTUNE daughterboard
        self.sma = SMAConnector()                       # SMA RF input

        # =====================================================================
        # Component Placement  (board is 200 x 150 mm, origin at centre)
        #
        #  Left  → power input / filtering
        #  Centre → three buck+LDO channels (top / middle / bottom)
        #  Right  → signal connectors & RF
        # =====================================================================

        # --- Input power (left edge) ---
        self.j_15v_in.at(-85, 0)                 # 15 V terminal block
        self.filter_15v.at(-65, 0)               # Input LC-pi filter
        self.j_dps5005_1.at(-85, 25)             # DPS5005 tap 1
        self.j_dps5005_2.at(-85, -25)            # DPS5005 tap 2

        # --- Channel 1: 15 V → 8.6 V → 7 V  (top row) ---
        self.buck_7v.at(-30, 45)                 # Buck 8.6 V
        self.ldo_7v.at(0, 45)                    # LDO 7 V

        # --- Channel 2: 15 V → 7 V → 5 V  (middle row, RedPitaya) ---
        self.buck_redpitaya.at(-30, 0)           # Buck 7 V
        self.ldo_redpitaya.at(0, 0)              # LDO 5 V
        self.filter_5v_rp.at(25, 0)              # Output LC-pi filter
        self.j_5v_redpitaya.at(45, 0)            # 5 V output connector

        # --- Channel 3: 15 V → 7 V → 5 V  (bottom row, VCO) ---
        self.buck_vco.at(-30, -45)               # Buck 7 V
        self.ldo_vco.at(0, -45)                  # LDO 5 V

        # --- Koheron laser driver (right-centre, tall footprint ~70 mm) ---
        self.koheron.at(55, 20)                  # Koheron CTL200

        # --- Signal connectors & RF (right edge) ---
        self.db15.at(85, 55)                     # DB15 connector (top-right)
        self.sma.at(85, -30)                     # SMA RF input
        self.bias_tee.at(65, -30)                # Bias tee
        self.rfenable.at(85, -50)                # RFENABLE header
        self.vtune_db.at(65, -50)                # VTUNE header

        # === Intermediate signal ports ===
        self.ld_p = Port()
        self.ld_n = Port()
        self.pelt_p = Port()
        self.pelt_n = Port()
        self.th_p = Port()
        self.th_n = Port()
        self.vtune = Port()
        self.vtune_mod = Port()
        self.vtune_modulated = Port()
        self.rf_enable = Port()

        # =====================================================================
        # Net list
        # =====================================================================
        self.nets = []

        # ----- 15 V input: connector → LC-pi filter → power_15v rail -----
        self.power_15v_raw = Power()
        self.nets += [
            self.j_15v_in.p1 + self.power_15v_raw.Vp,
            self.j_15v_in.p2 + self.power_15v_raw.Vn,
            # Filter input
            self.power_15v_raw.Vp + self.filter_15v.power_in.Vp,
            self.power_15v_raw.Vn + self.filter_15v.power_in.Vn,
            # Filter output → main 15 V rail
            self.filter_15v.power_out.Vp + self.power_15v.Vp,
            self.filter_15v.power_out.Vn + self.power_15v.Vn,
        ]

        # ----- DPS5005 output taps (from filtered 15 V) -----
        self.nets += [
            self.j_dps5005_1.p1 + self.power_15v.Vp,
            self.j_dps5005_1.p2 + self.power_15v.Vn,
            self.j_dps5005_2.p1 + self.power_15v.Vp,
            self.j_dps5005_2.p2 + self.power_15v.Vn,
        ]

        # ----- Channel 1: 15 V → 8.6 V (buck) → 7 V (LDO) -----
        self.nets += [
            # Buck input
            self.power_15v.Vp + self.buck_7v.power_in.Vp,
            self.power_15v.Vn + self.buck_7v.power_in.Vn,
            # Buck output → 8.6 V rail
            self.buck_7v.power_out.Vp + self.power_8v6.Vp,
            self.buck_7v.power_out.Vn + self.power_8v6.Vn,
            # LDO input
            self.power_8v6.Vp + self.ldo_7v.power_in.Vp,
            self.power_8v6.Vn + self.ldo_7v.power_in.Vn,
            # LDO output → 7 V rail
            self.ldo_7v.power_out.Vp + self.power_7v.Vp,
            self.ldo_7v.power_out.Vn + self.power_7v.Vn,
        ]

        # ----- Channel 2: 15 V → 7 V (buck) → 5 V (LDO) → filter -----
        self.nets += [
            self.power_15v.Vp + self.buck_redpitaya.power_in.Vp,
            self.power_15v.Vn + self.buck_redpitaya.power_in.Vn,
            self.buck_redpitaya.power_out.Vp + self.power_7v_redpitaya.Vp,
            self.buck_redpitaya.power_out.Vn + self.power_7v_redpitaya.Vn,
            self.power_7v_redpitaya.Vp + self.ldo_redpitaya.power_in.Vp,
            self.power_7v_redpitaya.Vn + self.ldo_redpitaya.power_in.Vn,
            self.ldo_redpitaya.power_out.Vp + self.power_5v_redpitaya.Vp,
            self.ldo_redpitaya.power_out.Vn + self.power_5v_redpitaya.Vn,
        ]

        # ----- Channel 3: 15 V → 7 V (buck) → 5 V (LDO) -----
        self.nets += [
            self.power_15v.Vp + self.buck_vco.power_in.Vp,
            self.power_15v.Vn + self.buck_vco.power_in.Vn,
            self.buck_vco.power_out.Vp + self.power_7v_vco.Vp,
            self.buck_vco.power_out.Vn + self.power_7v_vco.Vn,
            self.power_7v_vco.Vp + self.ldo_vco.power_in.Vp,
            self.power_7v_vco.Vn + self.ldo_vco.power_in.Vn,
            self.ldo_vco.power_out.Vp + self.power_5v_vco.Vp,
            self.ldo_vco.power_out.Vn + self.power_5v_vco.Vn,
        ]

        # ----- Koheron laser driver (powered from clean 7 V) -----
        self.nets += [
            self.power_7v.Vp + self.koheron.power.Vp,
            self.power_7v.Vn + self.koheron.power.Vn,
            self.koheron.ld_p + self.ld_p,
            self.koheron.ld_n + self.ld_n,
            self.koheron.tec_p + self.pelt_p,
            self.koheron.tec_n + self.pelt_n,
            self.koheron.th_p + self.th_p,
            self.koheron.th_n + self.th_n,
        ]

        # ----- 5 V RedPitaya output filter + connector -----
        self.power_5v_rp_filtered = Power()
        self.nets += [
            self.power_5v_redpitaya.Vp + self.filter_5v_rp.power_in.Vp,
            self.power_5v_redpitaya.Vn + self.filter_5v_rp.power_in.Vn,
            self.filter_5v_rp.power_out.Vp + self.power_5v_rp_filtered.Vp,
            self.filter_5v_rp.power_out.Vn + self.power_5v_rp_filtered.Vn,
            self.j_5v_redpitaya.p1 + self.power_5v_rp_filtered.Vp,
            self.j_5v_redpitaya.p2 + self.power_15v.Vn,  # GND from main rail
        ]

        # ----- DB15 connector -----
        self.nets += [
            self.db15.p1 + self.ld_p,            # Pin 1: LD+
            self.db15.p2 + self.pelt_n,           # Pin 2: Peltier −
            self.db15.p3 + self.pelt_p,           # Pin 3: Peltier +
            self.db15.p4 + self.th_p,             # Pin 4: Thermistor +
            self.db15.p5 + self.th_n,             # Pin 5: Thermistor −
            self.db15.p7 + self.power_5v_vco.Vp,  # Pin 7: 5 V VCO
            self.db15.p8 + self.vtune_modulated,  # Pin 8: VTUNE modulated
            self.db15.p9 + self.ld_n,             # Pin 9: LD−
            self.db15.p15 + self.rf_enable,       # Pin 15: RF enable
            self.db15.MH1 + self.power_15v.Vn,   # Mounting → GND
            self.db15.MH2 + self.power_15v.Vn,   # Mounting → GND
        ]

        # ----- RFENABLE daughterboard -----
        self.nets += [
            self.rfenable.p1 + self.rf_enable,
            self.rfenable.p2 + self.power_15v.Vn,
        ]

        # ----- VTUNE daughterboard -----
        self.nets += [
            self.vtune_db.p3 + self.vtune,
            self.vtune_db.p4 + self.power_15v.Vn,
        ]

        # ----- Bias tee (DC + RF combining for VCO tuning) -----
        self.nets += [
            self.bias_tee.dc_in + self.vtune,
            self.bias_tee.rf_in + self.vtune_mod,
            self.bias_tee.output + self.vtune_modulated,
        ]

        # ----- SMA connector (RF input for VCO modulation) -----
        self.nets += [
            self.sma.sig + self.vtune_mod,
            self.sma.gnd1 + self.power_15v.Vn,
            self.sma.gnd2 + self.power_15v.Vn,
            self.sma.gnd3 + self.power_15v.Vn,
            self.sma.gnd4 + self.power_15v.Vn,
        ]

        # ----- Unused pins -----
        # DB15 pins 6, 10-14 are spare
        self.db15.p6.no_connect()
        self.db15.p10.no_connect()
        self.db15.p11.no_connect()
        self.db15.p12.no_connect()
        self.db15.p13.no_connect()
        self.db15.p14.no_connect()

        # VTUNE daughterboard pins 1-2 unused
        self.vtune_db.p1.no_connect()
        self.vtune_db.p2.no_connect()

        # Buck enable/power-good and LDO enable are not connected at
        # the top level — they are pulled up internally via 10 kΩ resistors
        # in the TPSM863252 and NCP5663 sub-circuits.


# =============================================================================
# Design Entry Point
# =============================================================================

class OpenQuantumV5Design(Design):
    """Top-level JITX design for the OpenQuantum V5 board."""
    board = OQ5Board()
    substrate = OQ5Substrate()
    circuit = OpenQuantumV5()
