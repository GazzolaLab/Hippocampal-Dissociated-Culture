from neuron import h
import math
import numpy as np

class TomkoPyramidalCell:
    def __init__(self):
        # Create sections
        self.soma = h.Section(name='soma')
        self.radTprox1 = h.Section(name='radTprox1')
        self.radTprox2 = h.Section(name='radTprox2')
        self.radTmed1 = h.Section(name='radTmed1')
        self.radTmed2 = h.Section(name='radTmed2')
        self.radTdist1 = h.Section(name='radTdist1')
        self.radTdist2 = h.Section(name='radTdist2')
        self.rad_t1 = h.Section(name='rad_t1')
        self.rad_t2 = h.Section(name='rad_t2')
        self.rad_t3 = h.Section(name='rad_t3')
        self.lm_thick2 = h.Section(name='lm_thick2')
        self.lm_medium2 = h.Section(name='lm_medium2')
        self.lm_thin2 = h.Section(name='lm_thin2')
        self.lm_thick1 = h.Section(name='lm_thick1')
        self.lm_medium1 = h.Section(name='lm_medium1')
        self.lm_thin1 = h.Section(name='lm_thin1')
        self.oriprox1 = h.Section(name='oriprox1')
        self.oridist1_1 = h.Section(name='oridist1_1')
        self.oridist1_2 = h.Section(name='oridist1_2')
        self.oriprox2 = h.Section(name='oriprox2')
        self.oridist2_1 = h.Section(name='oridist2_1')
        self.oridist2_2 = h.Section(name='oridist2_2')
        self.axon = h.Section(name='axon')

        # Initialize all the section lists
        self.all = h.SectionList()
        self.soma_list = h.SectionList()
        self.apical_list = h.SectionList()
        self.basal_list = h.SectionList()
        self.axon_list = h.SectionList()
        self.oblique_list = h.SectionList()
        self.trunk_list = h.SectionList()

        # Layer-specific section lists
        self.soma_SP_list = h.SectionList()
        self.apical_SR_list = h.SectionList()
        self.apical_SLM_list = h.SectionList()
        self.basal_SO_list = h.SectionList()
        self.axon_SO_list = h.SectionList()
        self.ais_SR_list = h.SectionList()
        self.ais_SP_list = h.SectionList()

        # Initialize the cell
        self.init()

    def init(self):
        self.topol()
        self.basic_shape()
        self.subsets()
        self.geom()
        self.geom_nseg()
        self.insert_mech()
        self.biophys()

    def topol(self):
        # Connect sections
        self.radTprox1.connect(self.soma(1))
        self.radTprox2.connect(self.radTprox1(1))
        self.radTmed1.connect(self.radTprox2(1))
        self.radTmed2.connect(self.radTmed1(1))
        self.radTdist1.connect(self.radTmed2(1))
        self.radTdist2.connect(self.radTdist1(1))

        self.rad_t1.connect(self.radTprox1(1))
        self.rad_t2.connect(self.radTmed1(1))
        self.rad_t3.connect(self.radTdist1(1))

        self.lm_thick2.connect(self.radTdist2(1))
        self.lm_medium2.connect(self.lm_thick2(1))
        self.lm_thin2.connect(self.lm_medium2(1))
        self.lm_thick1.connect(self.radTdist2(1))
        self.lm_medium1.connect(self.lm_thick1(1))
        self.lm_thin1.connect(self.lm_medium1(1))

        self.oriprox1.connect(self.soma(0))
        self.oridist1_1.connect(self.oriprox1(1))
        self.oridist1_2.connect(self.oriprox1(1))
        self.oriprox2.connect(self.soma(1))
        self.oridist2_1.connect(self.oriprox2(1))
        self.oridist2_2.connect(self.oriprox2(1))
        self.axon.connect(self.soma(1))

    def basic_shape(self):
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0, 0, 0, 1, sec=self.soma)
        h.pt3dadd(15, 0, 0, 1, sec=self.soma)

        # Define 3D points for all sections
        section_points = {
            self.radTprox1: [(15, 0, 0, 1), (15, 15, 0, 1)],
            self.radTprox2: [(15, 15, 0, 1), (15, 30, 0, 1)],
            self.radTmed1: [(15, 30, 0, 1), (15, 45, 0, 1)],
            self.radTmed2: [(15, 45, 0, 1), (15, 60, 0, 1)],
            self.radTdist1: [(15, 60, 0, 1), (15, 75, 0, 1)],
            self.radTdist2: [(15, 75, 0, 1), (15, 90, 0, 1)],
            self.rad_t1: [(15, 15, 0, 1), (75, 45, 0, 1)],
            self.rad_t2: [(15, 45, 0, 1), (-45, 75, 0, 1)],
            self.rad_t3: [(15, 75, 0, 1), (75, 105, 0, 1)],
            self.lm_thick2: [(15, 90, 0, 1), (45, 105, 0, 1)],
            self.lm_medium2: [(45, 105, 0, 1), (75, 120, 0, 1)],
            self.lm_thin2: [(75, 120, 0, 1), (105, 135, 0, 1)],
            self.lm_thick1: [(15, 90, 0, 1), (-15, 105, 0, 1)],
            self.lm_medium1: [(-15, 105, 0, 1), (-45, 120, 0, 1)],
            self.lm_thin1: [(-45, 120, 0, 1), (-70, 135, 0, 1)],
            self.oriprox1: [(0, 0, 0, 1), (-45, -30, 0, 1)],
            self.oridist1_1: [(-45, -30, 0, 1), (-75, -60, 0, 1)],
            self.oridist1_2: [(-45, -30, 0, 1), (-85, -30, 0, 1)],
            self.oriprox2: [(15, 0, 0, 1), (60, -30, 0, 1)],
            self.oridist2_1: [(60, -30, 0, 1), (105, -60, 0, 1)],
            self.oridist2_2: [(60, -30, 0, 1), (100, -30, 0, 1)],
            self.axon: [(15, 0, 0, 1), (15, -150, 0, 1)]
        }

        for sec, points in section_points.items():
            h.pt3dclear(sec=sec)
            for point in points:
                h.pt3dadd(point[0], point[1], point[2], point[3], sec=sec)

    def subsets(self):
        # Add sections to all list
        for sec in [self.soma, self.radTprox1, self.radTprox2, self.radTmed1, self.radTmed2,
                    self.radTdist1, self.radTdist2, self.rad_t1, self.rad_t2, self.rad_t3,
                    self.lm_thick2, self.lm_medium2, self.lm_thin2, self.lm_thick1, self.lm_medium1,
                    self.lm_thin1, self.oriprox1, self.oridist1_1, self.oridist1_2, self.oriprox2,
                    self.oridist2_1, self.oridist2_2, self.axon]:
            self.all.append(sec=sec)

        # Apical dendrites
        for sec in [self.radTprox1, self.radTprox2, self.radTmed1, self.radTmed2,
                    self.radTdist1, self.radTdist2, self.rad_t1, self.rad_t2, self.rad_t3,
                    self.lm_thick2, self.lm_medium2, self.lm_thin2, self.lm_thick1,
                    self.lm_medium1, self.lm_thin1]:
            self.apical_list.append(sec=sec)

        # Initialize indices for different regions
        self.soma_SP_index = h.Vector()
        self.apical_SR_index = h.Vector()
        self.apical_SLM_index = h.Vector()
        self.basal_SO_index = h.Vector()
        self.axon_SO_index = h.Vector()
        self.ais_SR_index = h.Vector()
        self.ais_SP_index = h.Vector()

        # Add sections to specific lists with indices
        section_index = 0
        
        # Soma
        self.soma_list.append(sec=self.soma)
        self.soma_SP_list.append(sec=self.soma)
        self.soma_SP_index.append(section_index)
        section_index += 1

        # SR sections
        sr_sections = [self.radTprox1, self.radTprox2, self.radTmed1, self.radTmed2,
                       self.radTdist1, self.radTdist2, self.rad_t1, self.rad_t2, self.rad_t3]
        for sec in sr_sections:
            self.apical_SR_list.append(sec=sec)
            self.apical_SR_index.append(section_index)
            section_index += 1

        # SLM sections
        slm_sections = [self.lm_thick2, self.lm_medium2, self.lm_thin2,
                       self.lm_thick1, self.lm_medium1, self.lm_thin1]
        for sec in slm_sections:
            self.apical_SLM_list.append(sec=sec)
            self.apical_SLM_index.append(section_index)
            section_index += 1

        # Basal dendrites
        basal_sections = [self.oriprox1, self.oridist1_1, self.oridist1_2,
                         self.oriprox2, self.oridist2_1, self.oridist2_2]
        for sec in basal_sections:
            self.basal_list.append(sec=sec)
            self.basal_SO_list.append(sec=sec)
            self.basal_SO_index.append(section_index)
            section_index += 1

        # Axon
        self.axon_list.append(sec=self.axon)
        self.axon_SO_list.append(sec=self.axon)
        self.axon_SO_index.append(section_index)
        section_index += 1
        
        self.ais_SR_list.append(sec=self.axon)
        self.ais_SR_index.append(section_index)
        section_index += 1
        
        self.ais_SP_list.append(sec=self.axon)
        self.ais_SP_index.append(section_index)

        # Trunk list
        for sec in [self.radTprox1, self.radTprox2, self.radTmed1,
                    self.radTmed2, self.radTdist1, self.radTdist2]:
            self.trunk_list.append(sec=sec)

        # Oblique list
        for sec in [self.rad_t1, self.rad_t2, self.rad_t3]:
            self.oblique_list.append(sec=sec)

    def geom(self):
        # Set geometry parameters for all sections
        geom_params = {
            self.soma: (10, 10),
            self.radTprox1: (50, 4),
            self.radTprox2: (50, 4),
            self.radTmed1: (50, 3),
            self.radTmed2: (50, 3),
            self.radTdist1: (100, 2),
            self.radTdist2: (100, 2),
            self.rad_t1: (150, 1),
            self.rad_t2: (150, 1),
            self.rad_t3: (150, 1),
            self.lm_thick2: (100, 2),
            self.lm_medium2: (100, 1.5),
            self.lm_thin2: (50, 1),
            self.lm_thick1: (100, 2),
            self.lm_medium1: (100, 1.5),
            self.lm_thin1: (50, 1),
            self.oriprox1: (100, 2),
            self.oridist1_1: (200, 1.5),
            self.oridist1_2: (200, 1.5),
            self.oriprox2: (100, 2),
            self.oridist2_1: (200, 1.5),
            self.oridist2_2: (200, 1.5),
            self.axon: (150, 1)
        }

        for sec, (L, diam) in geom_params.items():
            sec.L = L
            sec.diam = diam

    def geom_nseg(self, freq=100, d_lambda=0.1):
        for sec in self.all:
            nseg = int((sec.L/(d_lambda*self.lambda_f(sec, freq))+0.9)/2)*2 + 1
            sec.nseg = nseg

    def lambda_f(self, section, freq):
        if section.n3d() < 2:
            return 1e5 * np.sqrt(section.diam/(4*np.pi*freq*section.Ra*section.cm))
        
        x1 = section.arc3d(0)
        d1 = section.diam3d(0)
        lam = 0
        
        for i in range(1, section.n3d()):
            x2 = section.arc3d(i)
            d2 = section.diam3d(i)
            lam += (x2 - x1)/np.sqrt(d1 + d2)
            x1, d1 = x2, d2
            
        lam *= np.sqrt(2) * 1e-5*np.sqrt(4*np.pi*freq*section.Ra*section.cm)
        return section.L/lam

    def distribute_distance(self, section_list, mechanism, expression):
        """
        Distribute mechanism values based on distance from soma.
        Args:
        section_list: NEURON SectionList object
        mechanism: String name of the mechanism
        expression: Expression for calculating values based on distance
        """
        h.distance(0, 0.5, sec=self.soma)  # Set soma as the origin
                                   
        for sec in section_list:
            for seg in sec:
                dist = h.distance(seg.x, sec=sec)
                mech_val = eval(expression % dist)  # Evaluate the expression with the distance
                setattr(seg, mechanism, mech_val)

    def insert_mech(self):

        for sec in self.all:
            sec.insert('pas')
            sec.insert('kdr')
            sec.insert('nax')
                                   
        for sec in self.soma_list:
            sec.insert('kmb')
            sec.insert('kap')
            sec.insert('hd')
            sec.insert('can')
            sec.insert('cal')
            sec.insert('cat')
            sec.insert('kca')
            sec.insert('cagk')
            sec.insert('cacum')
                                   
        for sec in self.apical_list:
            sec.insert('kad')
            sec.insert('hd')
            sec.insert('can')
            sec.insert('cal')
            sec.insert('cat')
            sec.insert('kca')
            sec.insert('cagk')
            sec.insert('cacum')

        for sec in self.basal_list:
            sec.insert('kad')
            sec.insert('hd')
            sec.insert('can')
            sec.insert('cal')
            sec.insert('cat')
            sec.insert('kca')
            sec.insert('cagk')
            sec.insert('cacum')

        for sec in self.axon_list:
            sec.insert('kmb')
            sec.insert('kap')

                                   
    def biophys(self):
        """Set biophysical properties of the cell."""
        # Set global parameters
        h.celsius = 35

        # Set parameters for all sections
        for sec in self.all:
            sec.cm = 1
            sec.ena = 50
            sec.ek = -90

        # Parameters for soma
        for sec in self.soma_list:
            sec.gkabar_kap = 0.0075
            sec.gbar_kmb = 0.001
            sec.gkdrbar_kdr = 0.0015
            sec.gbar_nax = 0.035
            sec.gcalbar_cal = 0.0005
            sec.gcanbar_can = 2.2618914062501833e-06
            sec.gcatbar_cat = 0.00005
            sec.gbar_kca = 0.0015
            sec.gbar_cagk = 4.4820097108998517e-05
            sec.Ra = 115.3957607556371
            sec.g_pas = 9.031387191839301e-05

        # Parameters for axon
        for sec in self.axon_list:
            sec.gbar_nax = 0.035
            sec.gkdrbar_kdr = 0.011664045469379856
            sec.gbar_kmb = 0.026473888790212396
            sec.gkabar_kap = 0.1636942175250268
            sec.Ra = 85.202399381150826
            sec.g_pas = 0.00012898002027660884
            sec.e_pas = -79.917091935442244

        # Parameters for apical dendrites
        for sec in self.apical_list:
            sec.gkdrbar_kdr = 0.0043036502438625682
            sec.gbar_nax = 0.038280628170345957
            sec.gcalbar_cal = 8.0324964335287e-06
            sec.gcanbar_can = 2.2618914062501833e-06
            sec.gcatbar_cat = 1.184948741542104e-06
            sec.gbar_kca = 9.0311387916396796e-05
            sec.gbar_cagk = 4.4820097108998517e-05
            sec.Ra = 115.3957607556371
            sec.g_pas = 9.031387191839301e-05

        # Parameters for trunk
        for sec in self.trunk_list:
            sec.gkdrbar_kdr = 0.02
            sec.gbar_nax = 0.025
            sec.gcalbar_cal = 8.0324964335287e-06
            sec.gcanbar_can = 2.2618914062501833e-06
            sec.gcatbar_cat = 1.184948741542104e-06
            sec.gbar_kca = 9.0311387916396796e-05
            sec.gbar_cagk = 4.4820097108998517e-05
            sec.Ra = 115.3957607556371
            sec.g_pas = 9.031387191839301e-05

        # Parameters for basal dendrites
        for sec in self.basal_list:
            sec.gkdrbar_kdr = 0.0043036502438625682
            sec.gbar_nax = 0.03
            sec.gcalbar_cal = 8.0324964335287e-06
            sec.gcanbar_can = 2.2618914062501833e-06
            sec.gcatbar_cat = 1.184948741542104e-06
            sec.gbar_kca = 9.0311387916396796e-05
            sec.gbar_cagk = 4.4820097108998517e-05
            sec.Ra = 115.3957607556371
            sec.g_pas = 9.031387191839301e-05

        # Distribute distance-dependent mechanisms
        # hd distribution
        self.distribute_distance(
            self.apical_list,
            "ghdbar_hd",
            "(1. + 3./100. * %.17g)*1.9042409723832741e-05"
        )

        # e_pas distribution
        self.distribute_distance(
            self.apical_list,
            "e_pas",
            "(-65.726902768520958-5*%.17g/150)"
        )

        # kad distribution
        self.distribute_distance(
            self.apical_list,
            "gkabar_kad",
            "(15./(1. + math.exp((300-%.17g)/50)))* 0.012921529390557651"
        )

        # Basal dendrites distributions
        self.distribute_distance(
            self.basal_list,
            "ghdbar_hd",
            "(1. + 3./100. * %.17g)*1.9042409723832741e-05"
        )
        self.distribute_distance(
            self.basal_list,
            "e_pas",
            "(-65.726902768520958-5*%.17g/150)"
        )
        self.distribute_distance(
            self.basal_list,
            "gkabar_kad",
            "(15./(1. + math.exp((300-%.17g)/50)))*0.012921529390557651"
        )

        # Soma distributions
        self.distribute_distance(
            self.soma_list,
            "ghdbar_hd",
            "(1. + 3./100. * %.17g)*1.9042409723832741e-05"
        )
        self.distribute_distance(
            self.soma_list,
            "e_pas",
            "(-65.726902768520958-5*%.17g/150)"
        )

        # Set specific kad values for certain sections
        self.radTprox1.gkabar_kad = 0.1
        self.radTprox2.gkabar_kad = 0.1
        self.rad_t1.gkabar_kad = 0.1
        self.radTmed1.gkabar_kad = 0.15
        self.radTmed2.gkabar_kad = 0.15

        # Special settings for rad_t2
        self.rad_t2.gkabar_kad = 0.1
        self.rad_t2.gbar_nax = 0.038
        self.rad_t2.gkdrbar_kdr = 0.002

        self.radTdist1.gkabar_kad = 0.2
        self.radTdist2.gkabar_kad = 0.2
        self.rad_t3.gkabar_kad = 0.25


    def position(self, x, y, z):
        xx = yy = zz = 0
        for sec in [self.soma, self.dend]:
            for i in range(sec.n3d()):
                pt3d = h.pt3dchange(i, 
                                  x - xx + sec.x3d(i),
                                  y - yy + sec.y3d(i),
                                  z - zz + sec.z3d(i),
                                  sec.diam3d(i))
        xx, yy, zz = x, y, z

    def is_art(self):
        return False

    def is_reduced(self):
        return True
