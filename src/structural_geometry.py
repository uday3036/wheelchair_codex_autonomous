"""Explicit connector envelopes. Visual only until hardware physics is qualified."""
import xml.etree.ElementTree as ET


def add_structural_geometry(root):
    def body(name):
        return root.find(f".//body[@name='{name}']")

    def shaft(owner, name, a, b, radius=.012):
        ET.SubElement(body(owner), 'geom', name='structure_'+name, type='capsule',
                      fromto=' '.join(map(str, (*a, *b))), size=str(radius),
                      mass='0', contype='0', conaffinity='0', group='3',
                      rgba='.48 .52 .55 1')

    def block(owner, name, pos, half):
        ET.SubElement(body(owner), 'geom', name='structure_'+name, type='box',
                      pos=' '.join(map(str, pos)), size=' '.join(map(str, half)),
                      mass='0', contype='0', conaffinity='0', group='3',
                      rgba='.48 .52 .55 1')

    # Chassis gaps found by the geometry-connectivity audit (see
    # docs/CONNECTOR_GEOMETRY.md).  Each block overlaps both neighbours by
    # 2.5-5 mm so joining never rests on a tangent face.
    # Core block top z=-0.02, deck box bottom z=+0.01 (30 mm gap): four standoffs.
    for i, (x, y) in enumerate([(-.06,-.14),(-.06,.14),(.10,-.14),(.10,.14)]):
        block('base_link', f'core_standoff_{i}', (x,y,-.005), (.018,.018,.0175))
    # Footrest hanger struts end 5 mm below the deck box: seat their tops.
    for side, sign in [('left', 1), ('right', -1)]:
        block('base_link', side+'_footrest_hanger', (.135,sign*.17,.006), (.022,.016,.008))
    # Left side plate (y 0.215-0.225) stands 25 mm off the core block wall (y 0.19).
    for i, z in enumerate((-.12, -.04)):
        block('base_link', f'left_side_plate_bracket_{i}', (-.07,.2025,z), (.02,.0175,.02))

    # With the rigid side-rail frame (frame_design.py) there are no rocker/bogie
    # bodies; the frame's own rails and arms (group 1) replace those connectors.
    linkage = body('rocker_left') is not None
    for side, sign in [('left', 1), ('right', -1)]:
        if linkage:
            shaft('base_link', side+'_rocker_spindle', (.02,sign*.16,0), (.02,sign*.30,0), .018)
        for kind, x, y, z_beam in [('front',.2,.17,-.175),('middle',-.185,.21,-.15)]:
            if linkage:
                # Riser starts inside the V-shaped bogie beam (beam centreline height
                # at this x is -0.19 front, -0.1535 middle) and rises to the support.
                shaft('bogie_'+side, side+'_'+kind+'_riser', (x,0,z_beam),(x,0,-.02), .016)
                shaft('bogie_'+side, side+'_'+kind+'_bridge', (x,0,-.02),(x,sign*y,-.02), .016)
            owner = 'front_'+side+'_knuckle' if kind=='front' else 'mid_'+side+'_caster_swivel'
            shaft(owner, side+'_'+kind+'_kingpin', (0,0,.135),(0,0,.19), .015)
            wheel_x=0 if kind=='front' else -.045
            shaft(owner, side+'_'+kind+'_axle', (wheel_x,-.075,0),(wheel_x,.075,0), .012)
        if linkage:
            shaft('rocker_'+side, side+'_rear_axle', (-.41,0,-.28),(-.41,sign*.14,-.28), .016)
            for kind, owner, x, z in [('rear','rocker_'+side,-.41,-.28),
                                       ('front','bogie_'+side,.2,-.19),
                                       ('mid','bogie_'+side,-.23,-.19)]:
                shaft(owner, side+'_'+kind+'_motor_bracket', (x,0,z),(x,sign*.05,z), .015)
        shaft('base_link', side+'_backrest_anchor', (-.12,sign*.16,.045),(-.14,sign*.195,.12))
    shaft('base_link','aux_mount', (.12,.17,.10),(.16,.26,.10), .022)
    # Rails stay below/outboard of the case interior and overlap sliding hood
    # walls throughout their travel; clearance is not hardware-certified.
    for kind,y,z,x0,x1 in [('main',.45,.14,-.45,.32),('aux',.323,.26,-.23,.40)]:
        if body(kind+'_case_lid') is not None:
            shaft('base_link',kind+'_cover_rail',(x0,y,z),(x1,y,z),.008)
            shaft('base_link',kind+'_rail_bracket',(.02,.19,.10),(.02,y,z),.010)
