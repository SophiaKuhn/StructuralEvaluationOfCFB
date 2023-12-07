import compas_fea.structure

WHITE_LIST_CLASS = [
    compas_fea.structure.Structure,
    compas_fea.structure.constraint.Constraint,
    compas_fea.structure.constraint.TieConstraint,
    compas_fea.structure.displacement.GeneralDisplacement,
    compas_fea.structure.displacement.FixedDisplacement,
    compas_fea.structure.displacement.PinnedDisplacement,
    compas_fea.structure.displacement.FixedDisplacementXX,
    compas_fea.structure.displacement.FixedDisplacementYY,
    compas_fea.structure.displacement.FixedDisplacementZZ,
    compas_fea.structure.displacement.RollerDisplacementX,
    compas_fea.structure.displacement.RollerDisplacementY,
    compas_fea.structure.displacement.RollerDisplacementZ,
    compas_fea.structure.displacement.RollerDisplacementXY,
    compas_fea.structure.displacement.RollerDisplacementYZ,
    compas_fea.structure.displacement.RollerDisplacementXZ,
    # from .element import (
    compas_fea.structure.element.Element,
    compas_fea.structure.element.BeamElement,
    compas_fea.structure.element.SpringElement,
    compas_fea.structure.element.TrussElement,
    compas_fea.structure.element.StrutElement,
    compas_fea.structure.element.TieElement,
    compas_fea.structure.element.ShellElement,
    compas_fea.structure.element.MembraneElement,
    compas_fea.structure.element.FaceElement,
    compas_fea.structure.element.SolidElement,
    compas_fea.structure.element.PentahedronElement,
    compas_fea.structure.element.TetrahedronElement,
    compas_fea.structure.element.HexahedronElement,
    compas_fea.structure.element.MassElement,
    # from .element_properties import ElementProperties
    compas_fea.structure.element_properties.ElementProperties,
    # from .interaction import Interaction, HeatTransfer
    compas_fea.structure.interaction.Interaction,
    compas_fea.structure.interaction.HeatTransfer,
    # from .load import (
    compas_fea.structure.load.Load,
    compas_fea.structure.load.PrestressLoad,
    compas_fea.structure.load.PointLoad,
    compas_fea.structure.load.PointLoads,
    compas_fea.structure.load.LineLoad,
    compas_fea.structure.load.AreaLoad,
    compas_fea.structure.load.GravityLoad,
    compas_fea.structure.load.ThermalLoad,
    compas_fea.structure.load.TributaryLoad,
    compas_fea.structure.load.HarmonicPointLoad,
    compas_fea.structure.load.HarmonicPressureLoad,
    compas_fea.structure.load.AcousticDiffuseFieldLoad,
    # )
    # from .material import (
    compas_fea.structure.material.Material,
    compas_fea.structure.material.Concrete,
    compas_fea.structure.material.ConcreteSmearedCrack,
    compas_fea.structure.material.ConcreteDamagedPlasticity,
    compas_fea.structure.material.ElasticIsotropic,
    compas_fea.structure.material.Stiff,
    compas_fea.structure.material.ElasticOrthotropic,
    compas_fea.structure.material.ElasticPlastic,
    compas_fea.structure.material.Steel,
    # )
    # from .misc import (
    compas_fea.structure.misc.Misc,
    compas_fea.structure.misc.Amplitude,
    compas_fea.structure.misc.Temperatures,
    # )
    # from .node import Node
    compas_fea.structure.node.Node,
    # from .section import (
    compas_fea.structure.section.Section,
    compas_fea.structure.section.AngleSection,
    compas_fea.structure.section.BoxSection,
    compas_fea.structure.section.CircularSection,
    compas_fea.structure.section.GeneralSection,
    compas_fea.structure.section.ISection,
    compas_fea.structure.section.PipeSection,
    compas_fea.structure.section.RectangularSection,
    compas_fea.structure.section.ShellSection,
    compas_fea.structure.section.MembraneSection,
    compas_fea.structure.section.SolidSection,
    compas_fea.structure.section.TrapezoidalSection,
    compas_fea.structure.section.TrussSection,
    compas_fea.structure.section.StrutSection,
    compas_fea.structure.section.TieSection,
    compas_fea.structure.section.SpringSection,
    compas_fea.structure.section.MassSection,
    # )
    # from .set import Set
    compas_fea.structure.set.Set,
    # from .step import (
    compas_fea.structure.step.Step,
    compas_fea.structure.step.GeneralStep,
    compas_fea.structure.step.ModalStep,
    compas_fea.structure.step.HarmonicStep,
    compas_fea.structure.step.BucklingStep,
    compas_fea.structure.step.AcousticStep,
    # )
    # from .structure import Structure
    compas_fea.structure.structure.Structure
]

FEA_WHITE_LIST = [c.__module__ + "." + c.__name__ for c in WHITE_LIST_CLASS]
