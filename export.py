import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from config import FlowstarConfig

"""This module contains helper classes to write SpaceEx XML files for a PWA and PWC neural abstraction, 
and flowstar files for Sigmoidal abstractions. It just keeps messy string formatting out of the neural abstraction class."""


def indent(elem: ET, level=0):
    """In-place pretty print of XML element

    Args:
        elem (ElementTree): XML element to pretty print
        level (int, optional): base indent level. Defaults to 0.
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class XMLWriter:
    """Helper class to write SpaceEx XML files for a PWA and PWC neural abstraction"""

    def __init__(self, abstraction):
        """Initialize XMLWriter

        Args:
            abstraction (NeuralAbstraction): Abstraction to write to XML
        """
        self.abstraction = abstraction
        self.dim = abstraction.dim
        self.locations = abstraction.locations
        self.invariants = abstraction.invariants
        self.flows = abstraction.flows
        self.modes = abstraction.modes
        self.error = abstraction.error
        self.transitions = abstraction.transitions

    def write(
        self,
        filename: str,
        bounded_time: bool = False,
        T: float = 1.0,
        initial_state=False,
    ):
        if initial_state:
            filename += "_init"
        SEP = " &\n"
        vx = ["x" + str(i) for i in range(self.dim)]
        vu = ["u" + str(i) for i in range(len(self.error))]
        var_attrib = {
            "name": None,
            "type": "real",
            "d1": "1",
            "d2": "1",
            "local": "false",
            "dynamics": "any",
            "controlled": "true",
        }
        root = ET.Element("xml")
        root = ET.Element(
            "sspaceex",
            {
                "xmlns": "http://www-verimag.imag.fr/xml-namespaces/sspaceex",
                "version": "0.2",
                "math": "SpaceEx",
            },
        )
        component = ET.SubElement(root, "component", {"id": "abstraction"})
        note = ET.SubElement(component, "note")
        # self.error = [0 for i in self.error]
        note.text = "Model error = {}".format(self.error)
        # Add variables  to component
        for var in vx:
            var_attrib.update({"name": var})
            ET.SubElement(component, "param", var_attrib)
        for var in vu:
            var_attrib.update({"name": var, "controlled": "false"})
            ET.SubElement(component, "param", var_attrib)
        if bounded_time:
            var_attrib.update({"name": "t", "controlled": "true"})
            ET.SubElement(component, "param", var_attrib)
            var_attrib.update({"name": "T", "dynamics": "const"})
            var_attrib.pop("controlled", None)
            ET.SubElement(component, "param", var_attrib)

        # Add each location (inv & flow)
        for loc_id in self.locations.keys():
            location = ET.SubElement(
                component, "location", {"id": loc_id, "name": "P" + loc_id}
            )

            ### Flows
            mode = self.modes[loc_id]
            flow_element = ET.SubElement(location, "flow")
            flow = self.flows[loc_id]
            flow_element.text = mode.flow_str(sep=" &\n")

            if bounded_time:
                flow_element.text += "&\nt'==1"

            ### Invariants
            inv = ET.SubElement(location, "invariant")
            P = self.invariants[loc_id]
            inv.text = mode.inv_str(sep=SEP)

            if bounded_time:
                inv.text += "&\nt <={}".format("T")

        if bounded_time:
            # Final location for bounded time
            location = ET.SubElement(
                component, "location", {"id": str(len(self.locations)), "name": "End"}
            )
            flow_element = ET.SubElement(location, "flow")
            flow_element.text = ""
            for i, var in enumerate(vx):
                flow_element.text += var + "'== 0" + SEP
            flow_element.text += "t'==0"
            inv = ET.SubElement(location, "invariant")
            inv.text = "t >={}\n".format("T")

        if initial_state:
            # Add extra location for initial state
            location = ET.SubElement(
                component,
                "location",
                {"id": str(len(self.locations) + 1), "name": "Init"},
            )
            flow_element = ET.SubElement(location, "flow")
            flow_element.text = ""
            for i, var in enumerate(vx):
                flow_element.text += var + "'== 0" + SEP
            flow_element.text = flow_element.text[:-2]
            if bounded_time:
                flow_element.text += SEP + "t'==1"
                # inv = ET.SubElement(location, 'invariant')
                # inv.text = "t == 0"

        # Add transitions between locations (guard, l0 l1)
        for transition in self.transitions:
            t0 = transition[0]
            t1 = transition[1]
            transition = ET.SubElement(
                component, "transition", {"source": str(t0), "target": str(t1)}
            )
            # ET.SubElement(transition, 'label').text = 'switch_mode'
            guard_element = ET.SubElement(transition, "guard")
            guard = self.invariants[t1]
            guard_element.text = guard.to_str(sep=SEP) + SEP
            if bounded_time:
                guard_element.text += "t <= {}&\n".format("T")
            guard_element.text = guard_element.text[:-2]

        if bounded_time:
            for loc_id in self.locations.keys():
                transition = ET.SubElement(
                    component,
                    "transition",
                    {"source": loc_id, "target": str(len(self.locations))},
                )
                guard_element = ET.SubElement(transition, "guard")
                guard_element.text = "t >= {}".format("T")

        if initial_state:
            for loc_id in self.locations.keys():
                mode = self.modes[loc_id]
                if mode.P.intersection(mode.P, initial_state).is_nonempty():
                    transition = ET.SubElement(
                        component,
                        "transition",
                        {"source": str(len(self.locations) + 1), "target": loc_id},
                    )

                    # guard_element = ET.SubElement(transition, 'guard')
                    # if bounded_time:
                    # guard_element.text +=  "t == 0"

        if bounded_time:
            # Now realise the component with value for T
            component = ET.SubElement(root, "component", {"id": "NA"})
            var_attrib = {
                "name": None,
                "type": "real",
                "d1": "1",
                "d2": "1",
                "local": "false",
                "dynamics": "any",
                "controlled": "true",
            }
            for var in vx:
                var_attrib.update({"name": var})
                ET.SubElement(component, "param", var_attrib)
            for var in vu:
                var_attrib.update({"name": var, "controlled": "false"})
                ET.SubElement(component, "param", var_attrib)
            var_attrib.update({"name": "t", "controlled": "true"})
            ET.SubElement(component, "param", var_attrib)
            bind = ET.SubElement(
                component, "bind", {"component": "abstraction", "as": "time_bounded"}
            )
            for var in vx:
                map = ET.SubElement(bind, "map", {"key": var})
                map.text = var
            for var in vu:
                map = ET.SubElement(bind, "map", {"key": var})
                map.text = var
            map = ET.SubElement(bind, "map", {"key": "t"})
            map.text = "t"
            map = ET.SubElement(bind, "map", {"key": "T"})
            map.text = T

        tree = ET.ElementTree(root)
        indent(root)
        tree.write("{}.xml".format(filename), encoding="", xml_declaration=True)


class FlowstarWriter:
    def __init__(self, abstraction):
        self.system = abstraction
        self.benchmark = abstraction.benchmark
        self.ndim = abstraction.ndim
        self.error = abstraction.error

    def write(self, filename: str, sym_N, initial: str, config: FlowstarConfig):
        vars = ["x{}".format(i) for i in range(self.ndim)]
        if len(vars) == 1:
            vars.insert(0, "t")
            sym_N = np.array([[1], [sym_N[0, 0]]])
            self.error = [0, self.error[0]]

        settings = self.get_settings(config, vars)
        part1 = (
            "continuous reachability\n{\nstate var "
            + ", ".join(vars)
            + "\n"
            + settings
            + "cutoff 1e-15\nprecision 32\noutput "
            + self.benchmark.short_name
            + "\nprint on\n}\nnonpoly ode\n{"
        )

        dyn = "\n"
        for i, xi in enumerate(vars):
            dyn += (
                "{}' = ".format(xi)
                + str(sym_N[i, 0])
                + " + [-{}, {}]".format(self.error[i], self.error[i])
                + "\n"
            )
        # dyn = "\nx1' = -x2 - 1.5 * x1^2 - 0.5 * x1^3 + [-0.1, 0.1] \nx2' = 3*x1 - x2 + [-0.1, 0.1]"
        part2 = "}\ninit\n{\n" + self.read_initial(initial) + "\n}\n}"
        model = part1 + dyn + part2
        with open(filename, "w") as fi:
            fi.write(model)

    def read_initial(self, initial):
        init = initial.replace(";", "\n")
        return init

    def get_settings(self, config, vars) -> str:
        text = "setting\n{\n"
        if config.step_mode == "adaptive":
            text += "adaptive steps {{min {}, max {}}}\n".format(
                config.step_size[0], config.step_size[1]
            )
        else:
            text += "fixed steps {}\n".format(config.step_size)
        text += "time {}\n".format(config.time)
        text += "remainder estimation {}\n".format(config.remainder_estimation)
        if config.qr_precondition:
            text += "QR precondition\n"
        else:
            text += "identity precondition\n"
        text += "gnuplot octagon "
        text += ", ".join(vars[:2])
        text += "\n"
        if config.order_mode == "fixed":
            text += "fixed orders {}\n".format(config.order)
        else:
            text += "adaptive orders {{min {}, max {}}}\n".format(
                config.order[0], config.order[1]
            )
        return text
