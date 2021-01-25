# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:29:03 2020

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import warnings

#%%
class SIR():
    """
    A CLASS FOR A SIMPLE SIR EPIDEMIOLOGICAL MODEL
    
        \dot{S} = -(1 - u) \\beta S I
        \dot{I} = (1 - u) \\beta S I - \gamma I
    
    THIS PARTICULAR CASE CONSIDERS A MAXIMUM POPULATION THAT CAN BE
    HOSPITALIZED AT THE SAME TIME, imax. THIS CLASS MAKES ALMOST NO USE OF
    NUMERICAL METHODS, GIVEN THAT MANY SOLUTIONS TO THE PRESENTED PROBLEMS CAN
    BE GIVEN ANALYTICALLY. HOWEVER, THE TIMES ASSOCIATED TO THE DIFFERENT
    TRAYECTORIES *ARE* CALCULATED NUMERICALLY.
    
    THE CLASS CONTAINS SEVERAL ATTRIBUTES AND METHODS.
    ATTRIBUTES:
        imax: FLOAT BETWEEN 0 AND 1. REPRESENTS THE FRACTION OF THE POPULATION
        THAT CAN BE IN MEDICAL FACILITIES AT THE SAME TIME. IN OTHER WORDS, A
        MEDICAL-FACILITY SATURATION POPULATION.
        umax: FLOAT BETWEEN 0 AND 1. THE INTENSITY OF THE CONTROL. IN THIS
        CASE, A DECREASE IN THE INFECTION RATE.
        gamma: HEALING RATE.
        beta: INFECTION RATE.
        sbar: A CRITICAL POPULATION DEFINED AS self.gamma/self.beta. WITH
        s < sbar, THE INFECTED POPULATION STARTS TO DECREASE INSTEAD OF
        INCREASE.
        sstar: SIMILAR TO sbar. INSTEAD OF BEING A CRITICAL POPULATION FOR THE
        NATURAL PROGRESION OF THE EPIDEMIC, IT IS A CRITICAL POPULATION FOR THE
        EPIDEMIC WITH u != 0.
        tau: A CURVE THAT REPRESENTS THE NATURAL PROGRESSION OF THE EPIDEMIC
        AND PASSES THROUGH self.imax AND self.sbar. ANY TRAJECTORY STARTING TO
        THE RIGHT OF THIS CURVE, IS IMMEDIATELY IN A "SAFE ZONE" OF "HERD
        IMMUNITY".
        phi: THE EQUIVALENT OF self.tau, BUT WITH u != self.umax. PASSES
        THROUGH self.sstar AND self.imax.
        theta: A SOLUTION TO THE DIFFERENTIAL EQUATION THAT PASSES THROUGH THE
        INTERSECTION OF self.tau WITH ZERO WITH u == self.umax.
        rho: A SOLUTION TO THE DIFFERENTIAL EQUATION WITH u == 0, THAT PASSES
        THROUGH self.sstar AND self.imax.
        points: A LIST OF Point OBJECTS INSIDE THE SYSTEM. THESE ARE USED TO 
        DETERMINE TRAJECTORIES.
    
    METHODS:
        set_params: A METHOD THAT SETS THE VALUES OF THE PARAMETERS. USES THE
        "HIDDEN" METHODS _set_imax, _set_umax, _set_gamma AND _set_beta.
        MODIFIES THE VALUES OF THE ATTRIBUTES self.imax, self.umax ET CETERA.
        _curve: A METHOD THAT CALCULATES THE I-VALUES OF THE S-VALUES GIVEN,
        WITH A GIVEN VALUE OF u.
        find_curves: FINDS AND SETS THE VALUES FOR self.tau, self.phi,
        self.theta AND self.rho. USES THE METHODS self.find_tau ET CETERA.
        add_point: ADDS A Point OBJECT TO self.points.
        find_regions: FINDS THE REGION IN WHICH EVERY POINT IN THE SYSTEM IS.
        get_trajectories: GETS A SAMPLE OF Trajectory OBJECTS FOR EVERY Point
        IN THE SYSTEM.
        get_shortest: GETS THE SHORTEST Trajectory (IN TIME, NOT LENGTH) FOR
        EACH OF THE Point OBJECTS IN THE SYSTEM.
    """
    params_set = False
    
    def __init__(self):
        self.params = None
        self.tau = None
        self.phi = None
        self.theta = None
        self.rho = None
        self.points = []
        self.commutation_curve = None
        
    
    def set_params(self, args, flag):
        """
        A FUNCTION THAT SETS THE PARAMETERS OF THE SYSTEM. ITS INPUTS ARE A
        LIST OR OTHER ITERABLE OBJECT WITH THE PARAMETER VALUES, AND A flag
        THAT DETERMINES WETHER THE RECEIVED VALUES ARE FOR beta AND gamma, OR
        R0 AND Rc.
        """
        self.params_set = True
        
        if flag == "bg":
            self._set_betagamma(args)
        elif flag == "r":
            self._set_r(args)
        else:
            print("Value type not recognized. "
                  "Flag has to be either 'bg' or 'r'.")
            return
        self.sbar = self.gamma/self.beta
        self.sstar = self.gamma / ((1 - self.umax)*self.beta)
        
        self._find_curves()
        
    
    def _set_r(self, args):
        """
        INPUT ARGUMENTS ARE Imax, umax and R0, IN THAT ORDER.
        """
        self.imax, self.umax, R0 = args
        self.gamma = 1 / 7
        self.beta = R0 * self.gamma
    
    
    def _set_betagamma(self, args):
        """
        INPUT ARGUMENTS ARE imax, umax, gamma AND beta, IN THAT ORDER.
        """
        self.imax, self.umax, self.gamma, self.beta = args

    
    def _curve(self, s = None, s_ref = None, i_ref = None, u = None):
        """
        A FUNCTION THAT RETURNS A CURVE OF PROGRESSION OF THE SYSTEM FROM A
        GIVEN INITIAL CONDITION.
        
        IN:
            s: VALUES AT WHICH THE CURVE IS TO BE EVALUATED.
            s_ref: A REFERENCE POINT FOR S. THE CURVE PASSES THROUGH THIS
            POINT.
            i_ref: A REFERENCE POINT FOR I. THE CURVE PASSES THROUGH THIS
            POINT.
            u: THE VALUE OF u. USUALLY, IT IS EITHER 0 OR self.umax.
        
        OUT:
            i_range: THE VALUES OF I FOR EACH VALUE OF s.
        """
        p1 = self.gamma / ((1 - u)*self.beta)
        p2 = np.log(s / s_ref)
        p3 = s - s_ref
        i_range = p1*p2 - p3 + i_ref
        return i_range
    
    
    def _find_tau(self):
        """
        A CurveSegment THAT PASSES THROUGH self.sbar AND self.imax WITH u == 0.
        THE CURVE IS THE LIMIT FOR THE SAFE ZONE.
        """
        s_range = np.linspace(max(1, self.phi.s[0]),
                              self.sbar,
                              10000, endpoint
                              = True)
        i_range = self._curve(s_range, self.sbar, self.imax, 0)
        self.tau = CurveSegment(s_range, i_range, 0, self)
    
    
    def _find_phi(self):
        """
        A CurveSegment THAT PASSES THROUGH self.sstar AND self.imax WITH
        u == self.umax.
        """
        s_range = np.linspace(2, 0, 10000, endpoint = False)
        i_range = self._curve(s_range, self.sstar, self.imax, self.umax)
        self.phi = CurveSegment(s_range, i_range, self.umax, self)
        new_endpoint = self.phi._curve_sol(0)[0]
        s_range = np.linspace(max(1, new_endpoint),
                              0,
                              10000,
                              endpoint = False)
        i_range = self._curve(s_range, self.sstar, self.imax, self.umax)
        self.phi = CurveSegment(s_range, i_range, self.umax, self)
    
    
    def _find_theta(self):
        """
        A CurveSegment THAT PASSES THROUGH THE INTERSECTION BETWEEN self.tau
        WITH 0, WITH u == self.umax. TRAJECTORIES THAT START BELOW THIS CURVE
        CANNOT REACH self.tau BY SETTING u == self.umax FROM THE BEGGINING.
        RATHER, THEY NEED TO GET OUT OF THAT REGION BY SETTING u == 0, AND
        ONLY THEN CAN THEY SET u == self.umax (OR AFTERWARDS).
        """
        s_init = self.tau.curve_intersection(self.phi)
        #print(s_init)
        s_zero = scipy.optimize.fsolve(self._curve, x0 = s_init[0],
                                       args = (self.sbar, self.imax, 0))
        #print(s_zero)
        s_range = np.linspace(max(1, self.phi.s[0]),
                              0,
                              10000,
                              endpoint = False)
        i_range = self._curve(s_range, s_zero, 0, self.umax)
        self.theta = CurveSegment(s_range, i_range, self.umax, self)
    
    
    def _find_rho(self):
        """
        A CurveSegment THAT PASSES THROUGH self.sstar AND self.imax.
        TRAJECTORIES STARTING TO THE RIGHT OF THIS CurveSegment WILL NOT BE
        ABLE TO REACH THE SINGULAR ARC DIRECTLY, BUT WILL HAVE TO GO THROUGH
        self.phi BEFORE.
        """
        s_range = np.linspace(max(1, self.phi.s[0]),
                              0,
                              10000,
                              endpoint = False)
        i_range = self._curve(s_range, self.sstar, self.imax, 0)
        self.rho = CurveSegment(s_range, i_range, 0, self)
    
    
    def _find_curves(self):
        self._find_phi()
        self._find_tau()
        self._find_theta()
        self._find_rho()
    
    
    def add_point(self, s0, i0):
        Px = Point(s0, i0)
        self.points.append(Px)
        return Px
    
    
    def find_region(self, p):
        idx_tau = np.searchsorted(np.flip(self.tau.s, 0), p.s0)
        idx_tau = len(self.tau.s) - idx_tau
        idx_phi = np.searchsorted(np.flip(self.phi.s, 0), p.s0)
        idx_phi = len(self.phi.s) - idx_phi
        idx_theta = np.searchsorted(np.flip(self.theta.s, 0), p.s0)
        idx_theta = len(self.theta.s) - idx_theta
        idx_rho = np.searchsorted(np.flip(self.rho.s, 0), p.s0)
        idx_rho = len(self.rho.s) - idx_rho
        if (p.s0 <= self.sbar
            or p.i0 <= self.tau.i[idx_tau]):
            p.region = 1
        elif (p.i0 > self.imax
              or (p.i0 > self.phi.i[idx_phi]
              and p.s0 > self.sstar)):
            p.region = 5
        elif (p.i0 <= self.rho.i[idx_rho]
              and p.i0 >= self.tau.i[idx_tau]
              and p.i0 >= self.theta.i[idx_theta]):
            p.region = 2
        elif (p.s0 >= self.sstar
              and p.i0 <= self.phi.i[idx_phi]
              and p.i0 >= self.theta.i[idx_theta]):
            p.region = 3
        elif (p.i0 < self.theta.i[idx_theta]):
            p.region = 4
    
    
    def find_regions(self):
        """
        A METHOD THAT DETERMINES THE REGION IN SPACE IN WHICH EACH Point OBJECT
        OF self.points IS. THE REGION IS DETERMINED BY THE CurveSegment OBECTS
        OF THE SYSTEM: self.tau, self.phi, self.theta AND self.rho. BRIEFLY:
            1: SAFE ZONE.
            2: CAN REACH self.tau BY STARTING WITH u == self.umax AT t == 0,
            REACHING THE SINGULAR ARC WITH u == 0 AND FOLLOWING IT, OR AT ANY
            INTERMEDIATE POINT.
            3: CAN REACH self.tau BY STARTING WITH u == self.umax AT t == 0,
            BUT HAS TO GO THROUGH self.phi BEFORE REACHING THE SINGULAR ARC.
            4: HAS TO SET  u == 0 UNTIL THE TRAJECTORY LEAVES REGION 4. THEN
            CAN REACH self.tau DEPENDING ON WHAT THE NEW REGION IS.
            5: CANNOT REACH self.tau WITHOUT EXCEEDING self.imax.
        """
        for p in self.points:
            if not p.region is None:
                pass
            else:
                self.find_region(p)
    
    
    def get_trajectories(self):
        for p in self.points:
            Tx = TrajectoryCreator(p, self)
            Tx.get_trajectories()
            p.trajectories = Tx.trajectories
    
    
    def get_shortest(self):
        for p in self.points:
            Mx = MinimumTrajectory(p, self)
            Mx.find_commutation()
            p.least_time = Mx.trajectory
    
    
    def remove_all_points(self):
        self.points = []

        
#%%
class PlotSIR():
    def __init__(self, obj):
        self.subject = obj
        self.fig, self.ax = plt.subplots()
    
    
    def show(self):
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, min(self.subject.imax * 1.1, 1))
        self.ax.set_xlabel("S")
        self.ax.set_ylabel("I")
                
        if self.subject.params_set:
            self.ax.plot(self.subject.sbar, self.subject.imax, "bx")
            self.ax.plot(self.subject.sstar, self.subject.imax, "rx")
            self.ax.plot([0, 1], [self.subject.imax]*2, "r--")
        
        self.ax.plot(self.subject.tau.s, self.subject.tau.i, "b-")
        self.ax.plot(self.subject.phi.s, self.subject.phi.i, "r-")
        self.ax.plot(self.subject.theta.s, self.subject.theta.i, "g-")
        self.ax.plot(self.subject.rho.s, self.subject.rho.i, "k-")


#%%
class Point():
    """
    A CLASS FOR A POINT IN A GIVEN SYSTEM.
    
    THE CLASS CONTAINS DIFFERENT ATTRIBUTES AND METHODS.
    ATTRIBUTES:
        s0: THE S COORDINATE OF THE POINT.
        i0: THE I COORDINATE OF THE POINT.
        state: AN ARRAY CONTAINING THE S AND I COORDINATES OF THE POINT.
        region: AN INTEGER DENOTING THE REGION IN WHICH THE POINT IS, RELATIVE
            TO THE CURVES OF THE SYSTEM THAT CONTAINS IT.
        trajectories: AN ARRAY OF TRAJECTORIES THAT GO FROM THE POINT TO ANY
            POINT OF THE "SAFE REGION" IN THE SYSTEM THAT CONTAINS IT. IF THE
            POINT IS IN REGION 5, ALL TRAJECTORIES ARE FauxTrajectories THAT
            SERVE AS PLACEHOLDERS FOR REAL TRAJECTORIES, SO THE CODE DOESN'T
            BREAK AT UNWANTED PLACES.
        times: AN ARRAY OF THE TOTAL TIME OF EACH OF THE TRAJECTORIES IN
            self.trajectories.
        i_times: AN ARRAY OF THE TIME OF EACH OF THE TRAJECTORIES IN 
            self.trajectories SINCE THE MOMENT OF INTERVENTION. THIS CAN BE A
            CurveSegment WITH u!=0, OR A LineSegment.
    METHODS:
        self.get_times(): A METHOD THAT CREATES THE ATTRIBUTE self.times.
        self.get_i_times(): A METHOD THAT CREATES THE ATTRIBUTE self.i_times.
    """
    def __init__(self, s0:float, i0:float):
        self.s0 = s0
        self.i0 = i0
        self.state = np.array([self.s0, self.i0]).reshape([2, 1])
        self.region = None
        self.trajectories = None
        self.least_time = None
    
    
    def __repr__(self):
        out = "Point object at (s, i) = ({:.2f}, {:.2f}).".format(self.s0,
                                                                  self.i0)
        return out
    
    
    def __print__(self):
        out = "Point object at (s, i) = ({:.2f}, {:.2f}).".format(self.s0,
                                                                  self.i0)
        return out
    
    
    def get_times(self):
        """
        A METHOD THAT CREATES THE ATTRIBUTE self.times. IF THE POINT IS LOCATED
        IN REGION 5, THIS RETURNS A None VALUE. ELSE, AN ARRAY WITH THE TOTAL
        TIME OF ALL THE TRAJECTORIES IS CREATED.
        """
        if self.region == 5:
            self.times = None
            return
        self.times = np.array([tra.get_time() for tra in self.trajectories])
    
    
    def get_i_times(self):
        """
        A METHOD THAT CREATES THE ATTRIBUTE self.i_times. IF THE POINT IS
        LOCATED IN REGION 5, THIS RETURNS A None VALUE. OTHERWISE, AN ARRAY
        WITH THE TIME SINCE INTERVENTION OF EVERY TRAJECTORY IN
        self.trajectories IS CREATED.
        """
        if self.region == 5:
            self.times = None
            return
        self.i_times = np.array([tra.get_intervention_time() for 
                                 tra in self.trajectories])
    
    
    def get_least_time(self):
        """
        A METHOD THAT RETURNS THE TRAJECTORY OF LEAST TOTAL TIME. IF THE POINT
        IS LOCATED IN REGION 5 OF THE SYSTEM, A FauxTrajectory IS RETURNED.
        OTHERWISE, THE RETURN VALUE IS THE TRAJECTORY OF LEAST TIME.
        """
        if self.region == 5:
            return self.trajectories[0]
        least_idx = np.where(self.times == min(self.times))
        return self.trajectories[least_idx][0]
    
    
    def get_least_intervention(self):
        """
        A METHOD THAT RETURNS THE TRAJECTORY OF LEAST INTERVENTION TIME. IF THE
        POINT IS LOCATED IN REGION 5 OF THE SYSTEM, A FauxTrajectory IS
        RETURNED. OTHERWISE, THE RETURN VALUE IS THE TRAJECTORY OF LEAST
        INTERVENTION TIME.
        """
        if self.region == 5:
            return self.trajectories[0]
        least_idx = np.where(self.i_times == min(self.i_times))
        return self.trajectories[least_idx][0]

    
    def minimize_time(self):
        """
        SOMETHING I'M STILL NOT VERY SURE HOW IT'S GOING TO WORK.
        """
    

#%%
class CurveSegment():
    """
    AN OBJECT THAT REPRESENTS A CURVE SEGMENT IN A GIVEN SYSTEM. IT IS
    INSTANTIATED WITH FOUR VARIABLES, PLUS TWO OPTIONAL.
    IN:
        s: EITHER FLOAT OF NP.ARRAY.
        i: EITHER FLOAT OF NP.ARRAY.
        u: FLOAT. DETERMINES THE VALUE OF THE CONTROL INPUT IN THE CURVE.
        system: THE SYSTEM TO WHICH THE CURVE BELONGS.
        s_end: THE ENDPOINT OF THE CURVE. IF NOT STATED, IT WILL DEFAULT TO 0.
            THIS "ENSURES" THAT CURVES ALWAYS MOVE TO THE LEFT.
        size: THE AMOUNT OF POINTS IN THE CURVE. DEFAULTS TO 50.
    
    THE OBJECT CONTAINS SEVERAL ATTRIBUTES AND METHODS.
    ATTRIBUTES:
        u: VALUE OF THE CONTROL INPUT.
        system: THE SYSTEM TO WHICH THE CurveSegment BELONGS.
        imax: THE VALUE OF THE MAXIMUM HOSPITAL CAPACITY IN A GIVEN SYSTEM.
            USED ONLY FOR CALCULATIONS.
        gamma: THE VALUE OF THE RATE OF RECOVERY IN THE INFECTED POPULATION.
            USED ONLY FOR CALCULATIONS.
        beta: THE VALUE OF THE RATE OF INFECTION. USED ONLY FOR CALCULATIONS.
        s: AN NP.ARRAY CONTAINING THE S-COORDINATES OF THE CURVE.
        i: AN NP.ARRAY CONTAINING THE I-COORDINATES OF THE CURVE.
        s_ref: A REFERENCE POINT WHERE THE CURVE PASSES IN S. USED MAINLY FOR
            CURVE CONSTRUCTION.
        i_ref: A REFERENCE POINT WHERE THE CURVE PASSES IN I. USED MAINLY FOR
            CURVE CONTSTRUCTION.
        time: TIME NEEDED TO GO FROM THE RIGHT TO THE LEFT SIDE OF THE CURVE.
        
    """
    def __init__(self, s, i, u:float, system:SIR, s_end = 0, size = 50):
        self.u = u
        self.ustart = self.u
        self.uend = self.u
        
        self.system = system
        self.imax = self.system.imax
        self.gamma = self.system.gamma
        self.beta = self.system.beta
        
        self.time = None

        if not isinstance(s, float) and not isinstance(i, float):
            self.s = s
            self.i = i
            self.s_ref = self.s[0]
            self.i_ref = self.i[0]
        else:
            self.s_ref = s
            self.i_ref = i
            if s_end == 0:
                self.s = np.linspace(self.s_ref, s_end, endpoint = False,
                                     num = size)
            else:
                self.s = np.linspace(self.s_ref, s_end, endpoint = True,
                                     num = size)
            self.i = self._curve(self.s, self.s_ref, self.i_ref)
    
    
    def __repr__(self):
        out = ("Curve segment:\n"
               "\tu = {:.2f}\n"
               "\ts_ref = {:.2f}\n"
               "\ti_ref = {:.2f}\n").format(self.u, self.s_ref, self.i_ref)
        if not self.time is None:
            out += "\tt = {:.2f}\n".format(self.time)
        return out


    def __print__(self):
        out = ("Curve segment:\n"
               "\tu = {:.2f}\n"
               "\ts_ref = {:.2f}\n"
               "\ti_ref = {:.2f}\n").format(self.u, self.s_ref, self.i_ref)
        if not self.time is None:
            out += "\tt = {:.2f}\n".format(self.time)
        return out
    
    
    def _curve(self, s, s_0 = None, i_0 = None, i_ref = 0):
        """
        A FUNCTION THAT RETURNS A CURVE OF PROGRESSION OF THE SYSTEM FROM A
        GIVEN INITIAL CONDITION.
        """
        if s_0 is None:
            s_0 = self.s_ref
        if i_0 is None:
            i_0 = self.i_ref
        p1 = self.gamma / ((1 - self.u)*self.beta)
        p2 = np.log(s / s_0)
        p3 = s - s_0
        i_range = (p1*p2) - p3 + i_0 - i_ref
        return i_range

  
    def _int(self, s):
        """
        A FUNCTION TO BE INTEGRATED, USED TO CALCULATE THE TIME TO GO FROM THE
        RIGHT TO THE LEFT SIDE OF THE CURVE.
        """
        return 1 / (s * self._curve(s))
    
    
    def _curve_sol(self, i_ref = 0):
        """
        A FUNCTION THAT RETURNS THE COORDINATES AT WHICH THE CURVE'S I-VALUE
        EQUALS i_ref.
        
        IN:
            i_ref: REFERENCE VALUE TO FIND SOLUTION FOR. IF NOT SPECIFIED,
            DEFAULTS TO 0.
        
        OUT:
            AN NP.ARRAY OF SHAPE [2, ] WITH THE S AND I COORDINATES OF THE
            INTERSECTION.
        """
        sol = scipy.optimize.fsolve(self._curve, self.s_ref,
                                    args = (None, None, i_ref))
        i = self._curve(sol)
        return np.array([sol, i]).reshape([2, ])
    
    
    def curve_intersection(self, other):
        """
        A FUNCTION THAT DETERMINES THE COORDINATES AT WHICH TWO CURVES
        INTERSECT.
        
        IN:
            other: ANOTHER CURVE OBJECT.
        
        OUT:
            AN NP.ARRAY OF SHAPE [2, ] WITH THE S AND I COORDINATES OF THE
            INTERSECTION.
        """
        denom = (1 / (1 - self.u)) - (1 / (1 - other.u))
        num = ((self.beta / self.gamma)*(other.s_ref - self.s_ref
                                         + other.i_ref - self.i_ref)
               + (1/(1 - self.u)*np.log(self.s_ref))
               - (1/(1 - other.u)*np.log(other.s_ref)))
        s_intersect = np.exp(num/denom)
        i_intersect = self._curve(s_intersect, self.s_ref, self.i_ref)
        return np.array([s_intersect, i_intersect])
    

    def get_time(self, start = None, end = None):
        """
        A METHOD THAT CREATES THE ATTRIBUTE self.time. IT INTEGRATES THE
        FUNCTION self._int TO CALCULATE THE TIME IT TAKES TO MOVE FROM THE
        RIGHT TO THE LEFT EXTREMES OF THE CURVE.
        
        IN:
            start: A STARTING POINT TO THE INTEGRATION. DEFAULTS TO THE FIRST
                S-COORDINATE IN THE TRAJECTORY.
            end: AN ENDING POINT TO THE INTEGRATION. DEFAULTS TO THE LAST
                S-COORDINATE IN THE TRAJECTORY.
            I SHOULD CONSIDER DELETING THESE INPUT VALUES, AS THERE IS REALLY
            NO POINT TO THEM. THERE IS, AS OF YET, NO REASON TO CHANGE THE
            INTEGRATION RANGE.
        """
        if start is None:
            start = self.s[0]
            end = self.s[-1]
        t, err = scipy.integrate.quad(self._int, start, end)
        t = -t / ((1 - self.u) * self.beta)
        self.time = t

        
#%%
class LineSegment():
    """
    AN OBJECT THAT REPRESENTS A LINE SEGMENT OF THE SINGULAR CURVE OF THE
    SYSTEM; A STRAIGHT LINE OF GRADIENT 0 THAT BETWEEN TWO DIFFERENT POINTS IN
    S, ALONG THE imax OF THE SYSTEM. IT IS INSTATIATED WITH THREE PARAMETERS,
    AND AN OPTIONAL ONE.
    IN:
        s_start: INITIAL POINT OF THE LINE.
        s_end: LAST POINT OF THE LINE.
        system: SYSTEM TO WHICH THE LINE BELONGS.
        size: AMOUNT OF POINTS ALONG THE LINE. DEFAULTS TO 50.
    
    THE OBJECT HAS SEVERAL ATTRIBUTES AND METHODS.
    ATTRIBUTES:
        system: THE SYSTEM THAT CONTAINS THE LINE SEGMENT.
        s: AN NP.ARRAY CONTAINING THE S-COORDINATES OF THE SEGMENT.
        i: THE I-COORDINATES OF THE SEGMENT. BASICALLY, AN NP.ARRAY OF size
            TIMES THE SYSTEM'S imax.
        sstart: INITIAL POINT OF THE LINE SEGMENT. USUALLY BIGGER THAN send.
        send: LAST POINT OF THE LINE SEGMENT. USUALLY SMALLER THAT sstart.
        time: THE AMOUNT OF ELAPSED TIME TO GO FROM sstart TO send ALONG THE
            LINE SEGMENT.
    METHODS:
        get_time: CREATES THE ATTRIBUTE self.time.
    """
    def __init__(self, s_start, s_end, i, system, size = 50):
        self.system = system
        self.s = np.linspace(s_start, s_end, num = size)
        self.i = np.array([i]*len(self.s))
        self.sstart = s_start
        self.send = s_end
        self.ustart = self.u_from_s(s_start)
        self.uend = self.u_from_s(s_end)
        self.time = None
        
    
    def __repr__(self):
        out = ("Line segment:\n"
               "\ts0 = {:.2f}\n"
               "\tsf = {:.2f}\n"
               "\ti = {:.2f}\n"
               "\tu = {:.2} to {:.2}\n").format(self.sstart, self.send,
                                                self.i[0], self.ustart,
                                                self.uend)
        if not self.time is None:
            out += "\tt = {:.2f}\n".format(self.time)
        return out
    
    
    def __print__(self):
        out = ("Line segment:\n"
               "\ts0 = {:.2f}\n"
               "\tsf = {:.2f}\n"
               "\ti = {:.2f}\n"
               "\tu = {:.2} to {:.2}\n").format(self.sstart, self.send,
                                                self.i[0], self.ustart,
                                                self.uend)
        if not self.time is None:
            out += "\tt = {:.2f}\n".format(self.time)
        return out
    
    
    def get_time(self, start = None, end = None):
        """
        PRETTY SELF-EXPLANATORY. CALCULATES THE TIME NEEDED TO GO FROM start
        TO end ALONG THE LINE SEGMENT.
        
        IN:
            start: THE INITIAL POINT FOR CALCULATION. DEFAULTS TO self.sstart.
            end: THE LAST POINT FOR CALCULATION. DEFAULTS TO self.send.
            I SHOULD CONSIDER DELETING THESE INPUT VARIABLES AS THEY REALLY
            DON'T CONTRIBUTE MUCH IN THE REAL USE OF THE OBJECT.
        """
        if start is None:
            start = self.sstart
        if end is None:
            end = self.send
        num = start - end
        denom = self.system.gamma * self.system.imax
        self.time = num / denom
    
    
    def u_from_s(self, s):
        """
        A FUNCTION TO DETERMINE THE VALUE OF u FOR A GIVEN s-COORDINATE ALONG
        THE SINGULAR ARC OF THE SYSTEM.
        
        IN:
            s: s-COORDINATE FOR WHICH THE VALUE OF u NEEDS TO BE FOUND.
        
        OUT:
            VALUE OF u FOR THE GIVEN s-COORDINATE.
        """
        m = self.system.umax / (self.system.sstar - self.system.sbar)
        b = ((self.system.sbar * self.system.umax)
             / (self.system.sstar - self.system.sbar))
        return m*s - b
    

#%%
class SingularCurve(LineSegment):
    """
    A CLASS THAT REPRESENTS THE SINGULAR CURVE.
    """
    def __init__(self, s_start, s_end, system, size = 50):
        self.system = system
        self.s = np.linspace(s_start, s_end, num = size)
        self.i = np.array([self.system.imax]*len(self.s))
        self.sstart = s_start
        self.send = s_end
        self.ustart = self.u_from_s(s_start)
        self.uend = self.u_from_s(s_end)
        self.time = None
    
    
    def __repr__(self):
        out = ("Singular Curve:\n"
               "\ts0 = {:.2f}\n"
               "\tsf = {:.2f}\n"
               "\tu = {:.2} to {:.2}\n").format(self.sstart, self.send,
                                                self.ustart, self.uend)
        if not self.time is None:
            out += "\tt = {:.2f}\n".format(self.time)
        return out
    
    
    def __print__(self):
        out = ("Singular Curve:\n"
               "\ts0 = {:.2f}\n"
               "\tsf = {:.2f}\n"
               "\tu = {:.2} to {:.2}\n").format(self.sstart, self.send,
                                                self.ustart, self.uend)
        if not self.time is None:
            out += "\tt = {:.2f}\n".format(self.time)
        return out
    
    
    def get_time(self, start = None, end = None):
        """
        PRETTY SELF-EXPLANATORY. CALCULATES THE TIME NEEDED TO GO FROM start
            TO end ALONG THE LINE SEGMENT.
        
        IN:
            start: THE INITIAL POINT FOR CALCULATION. DEFAULTS TO self.sstart.
            end: THE LAST POINT FOR CALCULATION. DEFAULTS TO self.send.
            I SHOULD CONSIDER DELETING THESE INPUT VARIABLES AS THEY REALLY
            DON'T CONTRIBUTE MUCH IN THE REAL USE OF THE OBJECT.
        """
        if start is None:
            start = self.sstart
        if end is None:
            end = self.send
        num = start - end
        denom = self.system.gamma * self.system.imax
        self.time = num / denom
    
    
    def u_from_s(self, s):
        """
        A FUNCTION TO DETERMINE THE VALUE OF u FOR A GIVEN s-COORDINATE ALONG
        THE SINGULAR ARC OF THE SYSTEM.
        
        IN:
            s: s-COORDINATE FOR WHICH THE VALUE OF u NEEDS TO BE FOUND.
        
        OUT:
            VALUE OF u FOR THE GIVEN s-COORDINATE.
        """
        m = self.system.umax / (self.system.sstar - self.system.sbar)
        b = ((self.system.sbar * self.system.umax)
             / (self.system.sstar - self.system.sbar))
        return m*s - b
    
    

#%%        
class TrajectoryCreator():
    """
    AN OBJECT THAT CREATES A SAMPLE OF PATHS FROM AN INITIAL POINT IN THE
    SYSTEM TO DIFFERENT POINTS IN SIR.tau. IT IS INSTATIATED WITH AN OBJECT
    OF CLASS Point, AND AN OBJECT OF CLASS SIR.
    
    THE CLASS HAS SEVERAL ATTRIBUTES AND METHODS
    ATTRIBUTES:
        point: THE POINT AT WHICH THE TRAJECTORY STARTS.
        region: THE REGION TO WHICH THAT POINT BELONGS. DEPENDING ON THE
            REGION, THE METHOD FOR CREATING THE TRAJECTORY VARIES.
        system: THE SYSTEM TO WHICH THE TRAJECTORY BELONGS.
        curve_u0: A CURVE THAT GOES FROM THE INITIAL POINT UNTIL ITS
            INTERSECTION WITH imax. ONLY CREATED IN CERTAIN CONIDTIONS.
        curve_umax: A CURVE THAT GOES FROM THE INITIAL POINT UNTIL ITS
            INTERSECTION WITH system.tau. ONLY CREATED IN CERTAIN CONDITIONS.
        u0_imax_intersection: A [2, ] NP.ARRAY. ITS CONTENTS ARE SELF-
            EXPLANATORY.
        umax_tau_intersection: A [2, ] NP.ARRAY. ITS CONTENTS ARE SELF-
            EXPLANATORY.
        shortest: THE TRAJECTORY OF LEAST TIME.
        
    """
    def __init__(self, Point:Point, system:SIR):
        self.point = Point
        self.region = Point.region
        self.system = system
        
        if self.region == 1:
            pass
        elif self.region == 2:
            self.curve_u0 = CurveSegment(self.point.s0, self.point.i0,
                                         0, self.system)
            self.u0_imax_intersection = self.curve_u0._curve_sol(self.system.imax)
            self.curve_u0 = CurveSegment(self.point.s0, self.point.i0, 0,
                                         self.system, self.u0_imax_intersection[0])
            self.curve_umax = CurveSegment(self.point.s0, self.point.i0,
                                           self.system.umax, self.system)
            self.umax_tau_intersection = self.curve_umax.curve_intersection(self.system.tau)
            self.curve_umax = CurveSegment(self.point.s0, self.point.i0,
                                           self.system.umax, self.system,
                                           self.umax_tau_intersection[0])
        elif self.region == 3:
            self.curve_u0 = CurveSegment(self.point.s0, self.point.i0,
                                         0, self.system)
            self.u0_imax_intersection = self.curve_u0._curve_sol(self.system.imax)
            self.curve_u0 = CurveSegment(self.point.s0, self.point.i0, 0,
                                         self.system, self.u0_imax_intersection[0])
            self.curve_umax = CurveSegment(self.point.s0, self.point.i0,
                                           self.system.umax, self.system)
            self.umax_tau_intersection = self.curve_umax.curve_intersection(self.system.tau)
            self.curve_umax = CurveSegment(self.point.s0, self.point.i0,
                                           self.system.umax, self.system,
                                           self.umax_tau_intersection[0])
        elif self.region == 4:
            self.curve_u0 = CurveSegment(self.point.s0, self.point.i0,
                                         0, self.system)
            self.curve_umax = CurveSegment(self.point.s0, self.point.i0,
                                           self.system.umax, self.system)
        elif self.region == 5:
            pass
    
    
    def _create_branch(self, point):
        """
        A METHOD THAT CREATES A CURVE THAT GOES FROM A GIVEN POINT IN A
        TRAJECTORY TO THE INTERSECTION WITH tau.
        """
        C = CurveSegment(point.s0, point.i0, self.system.umax, self.system)
        intersection = C.curve_intersection(self.system.tau)
        C = CurveSegment(point.s0, point.i0, self.system.umax, self.system,
                         intersection[0])
        return C
    
    
    def _method_region_1(self):
        """
        METHOD FOR CALCULATING THE PATHS FOR ANY POINT IN REGION 1.
        """
        Cx = CurveSegment(self.point.s0, self.point.i0, 0, self.system,
                          self.point.s0)
        Tx = Trajectory(Cx)
        self.trajectories = np.array([Tx])
        return self.trajectories
    

    def _method_region_2(self):
        """
        METHOD FOR CALCULATING THE PATHS FOR ANY POINT IN REGION 2.
        """
        #print("Method 2.")
        self.u0_imax_intersection = self.curve_u0._curve_sol(self.system.imax)
        self.C1 = CurveSegment(self.point.s0,
                               self.point.i0,
                               0,
                               self.system,
                               self.u0_imax_intersection[0])
        self.C2 = SingularCurve(self.u0_imax_intersection[0],
                                self.system.sbar,
                                self.system)
        self.main_trajectory = Trajectory(self.C1, self.C2)
        
        self.trajectories = np.empty([len(self.main_trajectory.s), ],
                                     dtype = object)
        self.trajectories[0] = Trajectory(self.curve_umax)
        for point in range(1, len(self.C1.s)):
            Px = Point(self.C1.s[point],
                       self.C1.i[point])
            Cx = self._create_branch(Px)
            Tx = Trajectory(CurveSegment(self.C1.s_ref,
                                         self.C1.i_ref,
                                         0,
                                         self.system,
                                         self.C1.s[point]),
                            Cx)
            self.trajectories[point] = Tx
        for point in range(len(self.C2.s)):
            Px = Point(self.C2.s[point],
                       self.C2.i[point])
            Cx = self._create_branch(Px)
            Tx = Trajectory(self.C1,
                            SingularCurve(self.C2.s[0],
                                          self.C2.s[point],
                                          self.system),
                            Cx)
            self.trajectories[point + len(self.C1.s)] = Tx
        return self.trajectories


    def _method_region_3(self):
        """
        METHOD FOR CALCULATING THE PATHS FOR ANY POINT IN REGION 3.
        """
        #print("Method 3.")
        self.u0_phi_intersection = self.curve_u0.curve_intersection(self.system.phi)
        self.C1 = CurveSegment(self.point.s0,
                               self.point.i0,
                               0,
                               self.system,
                               self.u0_phi_intersection[0])
        self.C2 = CurveSegment(self.u0_phi_intersection[0],
                               self.u0_phi_intersection[1],
                               self.system.umax,
                               self.system,
                               self.system.sstar)
        self.C3 = SingularCurve(self.system.sstar,
                                self.system.sbar,
                                self.system)
        self.main_trajectory = Trajectory(self.C1, self.C2, self.C3)
        
        self.trajectories = np.empty([len(self.C1.s) + len(self.C2.s),],
                                      dtype = object)
        self.trajectories[0] = Trajectory(self.curve_umax)
        for point in range(1, len(self.C1.s)):
            Px = Point(self.C1.s[point],
                       self.C1.i[point])
            Cx = self._create_branch(Px)
            Tx = Trajectory(CurveSegment(self.C1.s_ref,
                                         self.C1.i_ref,
                                         0,
                                         self.system,
                                         self.C1.s[point]),
                            Cx)
            self.trajectories[point] = Tx
        for point in range(len(self.C3.s)):
            Px = Point(self.C3.s[point],
                       self.C3.i[point])
            Cx = self._create_branch(Px)
            Tx = Trajectory(self.C1,
                            self.C2,
                            SingularCurve(self.system.sstar,
                                          self.C3.s[point],
                                          self.system),
                            Cx)
            self.trajectories[point + len(self.C2.s)] = Tx
        return self.trajectories
        

    def _method_region_4(self):
        """
        METHOD FOR CALCULATING THE PATHS FOR ANY POINT IN REGION 4.
        """
        #print("Method 4.")
        self.u0_theta_intersection = self.curve_u0.curve_intersection(self.system.theta)
        C1 = CurveSegment(self.point.s0,
                          self.point.i0,
                          0,
                          self.system,
                          self.u0_theta_intersection[0])
        Px = self.system.add_point(C1.s[-1], C1.i[-1]+1e-5)
        self.system.find_regions()
        Tx = TrajectoryCreator(Px, self.system)
        Tx.get_trajectories()
        if Px.region == 5:
            return
        self.trajectories = Tx.trajectories
        for trajectory in self.trajectories:
            trajectory.add_segment(C1)
        self.system.points.remove(Px)
        return self.trajectories


    def _method_region_5(self):
        """
        METHOD FOR CALCULATING THE PATHS FOR ANY POINT IN REGION 5.
        """
        print("This point is outside the region that fullfills the desired",
              " requirements. Hence, no trajectories are available.")
        self.trajectories = np.array([FauxTrajectory()]*100)
        return self.trajectories


    def get_trajectories(self):
        self._methods = {1: self._method_region_1,
                         2: self._method_region_2,
                         3: self._method_region_3,
                         4: self._method_region_4,
                         5: self._method_region_5}
        self._f = self._methods[self.region]
        return self._f()
    
    
    def get_times(self):
        """
        A METHOD THAT CALCULATES THE TIME FOR A CERTAIN TRAJECTORY. CALLS UPON
        THE METHODS OF TIME CALCULATION OF THE INDIVIDUAL SEGMENTS.
        """
        if self.point.region == 1 or self.point.region == 5:
            self._times = None
        else:
            for trajectory in self.trajectories:
                trajectory.get_time()
                self._times = np.array([tra.time for tra in self.trajectories])
                return self._times
        

#%%
class Trajectory():
    """
    AN INDIVIDUAL TRAJECTORY FROM AN INITIAL POINT TO A SAFE REGION. THIS
    CLASS HAS NO NOTION OF WHICH POINT OR SYSTEM IT BELONGS TO. IT JUST IS A
    SET OF SEGMENTS (EITHER CurveSegment or LineSegment).
    """
    def __init__(self, *args):
        self.segments = args
        self.s = np.concatenate([j.s for j in self.segments])
        self.i = np.concatenate([j.i for j in self.segments])
        self.time = None
        self.i_time = None
    
    
    def get_time(self, start = 0):
        """
        CALCULATES THE TIME FOR EACH SEGMENT IN THE TRAJECTORY, AND ASSIGNS THE
        VALUE TO self.time.
        """
        for segment in self.segments:
            segment.get_time()
        self.time = sum([segment.time for segment in self.segments])
        return self.time
    
    
    def get_intervention_time(self):
        """
        DETERMINES AT WHICH POINT IN THE TRAJECTORY u != 0. THEN ADDS THE TIMES
        FROM THAT POINT ONWARD. ASSIGNS THAT VALUE TO self.i_time.
        """
        io = self.find_intervention_onset()
        intervened = self.segments[io:]
        self.i_time = sum([segment.time for segment in intervened])
        return self.i_time
        
    
    def add_segment(self, segment, pos = "start"):
        """
        ADDS A SEGMENT AT THE START (DEFAULT) OR END OF A TRAJECTORY. MODIFIES
        THE ATTRIBUTE self.segments.
        
        IN:
            segment: AN OBJECT OF CLASS CurveSegment OR LineSegment.
            pos: ["start"/"end"] THE POSITION AT WHICH THE SEGMENT WILL BE
                ADDED.
        """
        if pos == "start":
            self.segments = (segment, *self.segments[:])
        elif pos == "end":
            self.segments = (*self.segments[:], segment)
        self.s = np.concatenate([j.s for j in self.segments])
        self.i = np.concatenate([j.i for j in self.segments])
    
    
    def find_intervention_onset(self):
        """
        RETURNS THE FIRST INSTANCE, IF ANY, IN WHICH THE TRAJECTORY HAS u != 0.
        THIS CAN BE EITHER WITH A CurveSegment WITH u != 0, OR ANY LineSegment,
        WHICH HAS A DIFFERENT VALUE OF u FOR EVERY POINT, BUT THEY ALL DIFFER
        FROM 0.
        """
        for ii in range(len(self.segments)):
            segment = self.segments[ii]
            #print(segment)
            if isinstance(segment, CurveSegment) and segment.u != 0:
                #print("Right Here!")
                return ii
            elif isinstance(segment, SingularCurve):
                #print("Right Here!")
                return ii
            else:
                #print("Not yet.")
                pass
        return None
    
    
    def plot_time(self):
        St = [s.time for s in self.segments]
        St = [[0, t] for t in St]
        #print(St)
        Su = [[s.ustart, s.uend] for s in self.segments]
        Su = [z for y in Su for z in y]
        #print(Su)
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$u$")
        ax.plot(np.cumsum(St), Su)


#%%
class FauxTrajectory():
    """
    A FAKE TRAJECTORY FOR POINTS IN REGION 5. THIS HAS "GHOST" ATTRIBUTES SUCH
    AS s, i, time AND i_time SIMILAR TO THOSE OF NORMAL TRAJECTORIES, EXCEPT
    THAT THESE ALL HOLD EITHER NP.NAN OF None VALUES. THIS EXISTS MERELY AS A 
    WAY TO BE ABLE TU USE THE SAME PROGRAM LOGIC WITHOUT PRIOR KNOWLEDGE OF THE
    POINT REGION.
    """
    def __init__(self):
        self.s = np.array([np.nan]*100)
        self.i = np.array([np.nan]*100)
        self.time = None
        self.i_time = None

    
#%%
class MinimumTrajectory():
    def __init__(self, point, system):
        self.point = point
        self.system = system
        self.trajectory = None
        self.commutation = None
    
    
    def find_commutation(self):
        """
        A FUNCTION THAT FINDS THE OPTIMUM COMMUTATION POINT BY MINIMIZING A
        TIME FUNCTION. WHICH TIME FUNCTION EXACTLY WILL DEPEND ON THE REGION
        THE POINT IS IN.
        """
        if self.point.region == 4:
            self._region4(self.point.s0)
            return
        if self.point.region == 5:
            #print("I gotta do something here.")
            self._region5(self.point.s0)
            return
        sol = scipy.optimize.minimize(self.ttime, self.point.s0,
                                      bounds = ((self.system.sbar,
                                                 self.point.s0), ))
        self.commutation = sol.x[0]
        self.trajectory = self.make(self.commutation)
        self.trajectory.get_time()
    
    
    def ttime(self, cp):
        """
        A FUNCTION THAT CREATES A TRAJECTORY AND RETURNS ITS TIME, DEPENDING
        ON A GIVEN COMMUTATION POINT. THIS FUNCTION IS MADE TO BE MINIMIZED AND
        GET THE IDEAL COMMUTATION POINT.
        """
        Tx = self.make(cp)
        Tx.get_time()
        return Tx.time
        
    
    def make(self, cp):
        """
        A FUNCTION THAT WILL CREATE THE LEAST-TIME TRAJECTORY WITH A GIVEN
        COMMUTATION POINT cp.
        """
        self._f = {1: self._region1,
                   2: self._region2,
                   3: self._region3,
                   4: self._region4,
                   5: self._region5
                   }
        self._f = self._f[self.point.region]
        return self._f(cp)
    
    
    def _region1(self, s_c):
        pass
    
    
    def _region2(self, s_c):
        """
        A FUNCTION THAT CREATES A TRAJECTORY IN REGION 2 OF THE SYSTEM, WITH
        THE INITIAL CONDITIONS AT THE Point COORDINATES AND u == 0. THE
        COMMUTATION OF THE TRAJECTORY IS GIVEN BY THE PARAMETER s_c.
        """
        try:
            s_c = s_c[0]
        except:
            pass
            #print("Input is already a float.")
        #print("The given commutation point is: {}".format(s_c))
        u0_curve = CurveSegment(self.point.s0, self.point.i0, 0,
                                self.system)
        sc, ic = u0_curve._curve_sol(self.system.imax)
        #print("The intersection point is: {}".format(sc))
        if s_c >= sc:
            #print("I'ma do it with only two thingamajigs.")
            Tu = CurveSegment(self.point.s0, self.point.i0, 0,
                              self.system, s_c)
            Tu.get_time()
            i_c = self.system._curve(s_c, self.point.s0, self.point.i0, 0)
            Tc = CurveSegment(s_c, i_c, self.system.umax, self.system)
            send, iend = Tc.curve_intersection(self.system.tau)
            Tc = CurveSegment(s_c, i_c, self.system.umax, self.system, send)
            Tc.get_time()
            #print("Tu: {}".format(Tu.time))
            #print("Tc: {}".format(Tc.time))
            #print(Tu.time + Tc.time)
            return Trajectory(Tu, Tc)
        else:
            #print("I'ma have to do it with three thingamajigs.")
            Tu = CurveSegment(self.point.s0, self.point.i0, 0,
                              self.system, sc)
            Tu.get_time()
            Ts = SingularCurve(sc, s_c, self.system)
            Ts.get_time()
            Tc = CurveSegment(s_c, self.system.imax, self.system.umax,
                              self.system)
            send, iend = Tc.curve_intersection(self.system.tau)
            Tc = CurveSegment(s_c, self.system.imax, self.system.umax,
                              self.system, send)
            Tc.get_time()
            #print("Tu: {}".format(Tu.time))
            #print("Ts: {}".format(Ts.time))
            #print("Tc: {}".format(Tc.time))
            #print(Tu.time + Ts.time + Tc.time)
            return Trajectory(Tu, Ts, Tc)


    def _region3(self, s_c):
        """
        A FUNCTION THAT CREATES A TRAJECTORY IN REGION 3 OF THE SYSTEM, WITH
        THE INITIAL CONDITIONS AT THE Point COORDINATES AND u == 0. THE
        COMMUTATION OF THE TRAJECTORY IS GIVEN BY THE PARAMETER s_c.
        """
        """
        CALCULATE THE INTERSECTION BETWEEN THE U0_CURVE AND PHI. THEN DO
        SOMETHING SIMILAR AS IN THE CASE OF REGION 2, DEPENDING ON WETHER OR
        NOT THE INTERSECTION IS BEFORE SYSTEM.SSTAR.
        """
        try:
            s_c = s_c[0]
        except:
            pass
            #print("Input is already a float.")
        u0_curve = CurveSegment(self.point.s0, self.point.i0,
                                0, self.system)
        #print(u0_curve.s)
        sc, ic = u0_curve.curve_intersection(self.system.phi)
        if s_c >= sc:
            Tu = CurveSegment(self.point.s0, self.point.i0, 0, self.system,
                              s_c)
            Tu.get_time()
            i_c = self.system._curve(s_c, self.point.s0, self.point.i0, 0)
            Tc = CurveSegment(s_c, i_c, self.system.umax, self.system)
            send, iend = Tc.curve_intersection(self.system.tau)
            Tc = CurveSegment(s_c, i_c, self.system.umax, self.system, send)
            Tc.get_time()
            return Trajectory(Tu, Tc)
        elif s_c < sc and s_c >= self.system.sstar:
            Tu = CurveSegment(self.point.s0, self.point.i0, 0, self.system,
                              sc)
            Tu.get_time()
            Tc = CurveSegment(sc, ic, self.system.umax, self.system)
            send, iend = Tc.curve_intersection(self.system.tau)
            Tc = CurveSegment(sc, ic, self.system.umax, self.system, send)
            Tc.get_time()
            return Trajectory(Tu, Tc)
        elif s_c < self.system.sstar:
            Tu = CurveSegment(self.point.s0, self.point.i0, 0, self.system,
                              sc)
            Tu.get_time()
            Tc1 = CurveSegment(sc, ic, self.system.umax, self.system,
                               self.system.sstar)
            Tc1.get_time()
            Ts = SingularCurve(self.system.sstar, s_c, self.system)
            Ts.get_time()
            Tc2 = CurveSegment(s_c, self.system.imax, self.system.umax,
                               self.system)
            send, iend = Tc2.curve_intersection(self.system.tau)
            Tc2 = CurveSegment(s_c, self.system.imax, self.system.umax,
                               self.system, send)
            Tc2.get_time()
            return Trajectory(Tu, Tc1, Ts, Tc2)


    def _region4(self, s_c):
        u0_curve = CurveSegment(self.point.s0, self.point.i0, 0, self.system)
        sc, ic = u0_curve.curve_intersection(self.system.theta)
        u0_curve = CurveSegment(self.point.s0, self.point.i0, 0, self.system,
                                sc)
        Px = self.system.add_point(u0_curve.s[-1], u0_curve.i[-1]+1e-3)
        self.system.find_regions()
        Mx = MinimumTrajectory(Px, self.system)
        Mx.find_commutation()
        Tx = Mx.trajectory
        Tx.add_segment(u0_curve)
        Tx.get_time()
        self.commutation = Mx.commutation
        self.trajectory = Tx
        self.system.points.remove(Px)


    def _region5(self, s_c):
        u0_curve = CurveSegment(self.point.s0, self.point.i0, self.system.umax,
                                self.system)
        sc, ic = u0_curve.curve_intersection(self.system.tau)
        send = min([sc, self.system.sbar])
        u0_curve = CurveSegment(self.point.s0, self.point.i0, self.system.umax,
                                self.system, send)
        self.trajectory = Trajectory(u0_curve)


#%%
class Comp():
    """
    A CLASS WHOSE SOLE PURPOSE IS TO COMPARE THE STUFF BETWEEN MODELS OF
    ESTIMATED PARAMETERS AND THEIR REAL COUNTERPARTS.
    """
    def __init__(self, real:SIR, estimated:SIR, point:Point):
        self.real = real
        self.estimated = estimated
        self.point = point
    
    
    def compare(self):
        """
        THIS FUNCTION WILL SERVE TO COMPARE THE "EFFICIENCY" OF THE MODELS WITH
        REAL AND ESTIMATED PARAMETERS.
        """
        self._rp =  self.real.add_point(self.point.s0, self.point.i0)
        self.real.find_regions()
        self.real.get_shortest()
        self._ep =  self.estimated.add_point(self.point.s0, self.point.i0)
        self.estimated.find_regions()
        self.estimated.get_shortest()
        self.ideal = self._rp.least_time
        self.reference = self._ep.least_time
        crits = [j.s[-1] for j in self.reference.segments]
        print(crits)
        self.imitation = self.imitate(crits)
        self.ratio = max(self.imitation.i) / self.real.imax
    
    
    def get_criticals(self):
        """
        THIS DETERMINES WHICH POINTS ARE THE STARTING AND FINAL POINTS FOR
        THE SEGMENTS.
        """
        return self._crits_3()
    
    
    def _crits_2(self):
        [a, b] = self.reference.segments
        c1 = a.s[-1]
        c2 = b.s[-1]
        return [c1, c2]
    
    
    def _crits_3(self):
        [a, b, c] = self.reference.segments
        c1 = a.s[-1]
        c2 = b.s[-1]
        c3 = c.s[-1]
        return [c1, c2, c3]
    
    
    def _crits_4(self):
        [a, b, c, d] = self.reference.segments
        c1 = a.s[-1]
        c2 = a.s[-1]
        return [c1, c2]
    
    
    def imitate(self, crits):
        """
        THIS CREATES THE "IMITATION" OF THE REFERENCE CURVE IN THE REAL-LIFE
        SYSTEM.
        """
        methods = {2: self._imitate_2,
                   3: self._imitate_3,
                   4: self._imitate_4}
        self.f = methods[len(self.reference.segments)]
        return self.f(crits)
    
    
    def _imitate_2(self, crits):
        C1 = CurveSegment(self.point.s0, self.point.i0, 0, self.real,
                          crits[0])
        C2 = CurveSegment(C1.s[-1], C1.i[-1], self.real.umax, self.real,
                          crits[1])
        FP = CurveSegment(C2.s[-1], C2.i[-1], 0, self.real, self.real.sbar)
        T = Trajectory(C1, C2, FP)
        return T


    def _imitate_3(self, crits):
        C1 = CurveSegment(self.point.s0, self.point.i0, 0, self.real,
                          crits[0])
        L = LineSegment(crits[0], crits[1], C1.i[-1], self.real)
        C2 = CurveSegment(L.s[-1], L.i[-1], self.real.umax, self.real,
                          crits[2])
        FP = CurveSegment(C2.s[-1], C2.i[-1], 0, self.real,
                          self.real.sbar)
        T = Trajectory(C1, L, C2, FP)
        return T


    def _imitate_4(self, crits):
        C1 = CurveSegment(self.point.s0, self.point.i0, 0, self.real,
                          crits[0])
        C2 = CurveSegment(C1.s[-1], C1.i[-1], self.real.umax, self.real,
                          crits[1])
        L = LineSegment(crits[1], crits[2], C2.i[-1], self.real)
        C3 = CurveSegment(L.s[-1], L.i[-1], self.real.umax, self.real,
                          crits[3])
        FP = CurveSegment(C3.s[-1], C3.i[-1], 0, self.real, self.real.sbar)
        T = Trajectory(C1, C2, L, C3, FP)
        return T


#%%
def create_initial_conditions(sys:SIR, displacement:float, i_low:float,
                              start:float = None, end:float = None,
                              size = 100):
    """
    A FUNCTION THAT CREATES THE INITIAL CONDITIONS FOR THE SEARCH OF THE
    COMMUTATION CURVE IN A DETERMINED SYSTEM. TAKES AS INPUT THE SYSTEM THAT
    IS GOING TO BE ANALYZED, THE displacement TO THE RIGHT, AND HOW FAR i_low
    THE LINE SEGMENT IS GOING TO GO.
    """
    if start is None:
        start = sys.sbar + displacement
    if end is None:
        end = 1 + displacement
    if start is None and end is None:
        print("This shouldn't happen anymore. Weird if it did.")
        s_inter, i_inter = sys.tau._curve_sol(i_low)
        C = CurveSegment(sys.sbar, sys.imax, 0, sys, s_inter,
                         size = int(size / 2))
        s0 = np.linspace(C.s[-1], 1 - displacement, num = int(size / 2))
        i0 = np.array([i_low] * len(s0))
        s0 = np.concatenate((C.s, s0))
        s0 = s0 + displacement
        i0 = np.concatenate((C.i, i0))
        return s0, i0
    else:
        #print("Aw lawd this is morE CoMpLIcAtED!")
        start -= displacement
        end -= displacement
        s_inter, i_inter = sys.tau._curve_sol(i_low)
        #print(start, end, s_inter, i_inter)
        if start < s_inter and end > s_inter:
            #print("Case 1: mixed bag.")
            istart = sys._curve(start, sys.sbar, sys.imax, 0)
            C = CurveSegment(start, istart, 0, sys, s_inter,
                             size = int(size / 2))
            s0 = np.linspace(C.s[-1], end, num = int(size / 2))
            i0 = np.array([i_low] * len(s0))
            s0 = np.concatenate((C.s, s0))
            s0 += displacement
            i0 = np.concatenate((C.i, i0))
            return s0, i0
        elif start < s_inter and end <= s_inter:
            #print("Case 2: all in curve.")
            istart = sys._curve(start, sys.sbar, sys.imax, 0)
            C = CurveSegment(start, istart, 0, sys, end)
            s0 = C.s + displacement
            i0 = C.i
            return s0, i0
        elif start >= s_inter and end > s_inter:
            #print("Case 3: all in straight.")
            s0 = np.linspace(start, end, num = size) + displacement
            i0 = np.array([i_low] * len(s0))
            return s0, i0
        else:
            print("Case 4: WTF is going on?!")
            return None
    

#%%
def find_relevant_change(array, err = 1e-3):
    """
    FIND THE POINT AT WHICH AN ARRAY STARTS CHANGING "SIGNIFICANTLY".
    """
    dif = np.abs(array - array[0])
    dif = dif > err
    idx = min(np.where(dif == True)[0])
    return idx


#%%
def find_max(trajectory):
    """
    FINDS THE COORDINATE OF THE MAXIMUM POINT IN A TRAJECTORY.
    """
    x = trajectory.s
    y = trajectory.i
    yt = np.abs(y - max(y))
    yt = yt < 1e-5
    max_idx = np.where(yt == True)[0]
    max_idx = max(max_idx)
    return [x[max_idx], y[max_idx]]


#%%
def find_criticals(s0, fp):
    fp_diff = np.diff(fp)
    crit1 = find_relevant_change(fp_diff)
    max_idx = max(np.where(fp == max(fp))[0])
    start = s0[crit1]
    #print(start)
    try:
        crit2 = min(np.where(fp_diff[max_idx:] > 0)[0]) + max_idx
        end = s0[crit2]
        reached = True
    except:
        warnings.warn(("The tail of the curve is not reachable. "
                       "The commutation curve will be incomplete."), Warning)
        end = s0[-1]
        reached = False
    return start, end, reached


#%%
def find_commutation_curve(system:SIR, displacement = 1e-3, ilow = 5e-4):
    """
    A function that takes a SIR system as an argument, and finds its
    commutation curve.
    """
    found = False
    endpoint = system.phi._curve_sol(ilow)[0]
    while not found:
        s0, i0 = create_initial_conditions(system, displacement, ilow,
                                           end = endpoint)
        for s, i in zip(s0, i0):
            system.add_point(s, i)
        
        system.find_regions()
        system.get_shortest()
        M = np.array([p.least_time for p in system.points])
        
        final_point = np.array([tra.s[-1] for tra in M])
        st, end, found = find_criticals(s0, final_point)
        forced = system.phi._curve_sol(ilow)
        last = max(forced)
        system.remove_all_points()
    
    s0, i0 = create_initial_conditions(system, displacement, ilow, start = st,
                                       end = end)
    system.remove_all_points()
    for s, i in zip(s0, i0):
        system.add_point(s, i)
    
    system.find_regions()
    system.get_shortest()
    M = np.array([p.least_time for p in system.points])
    
    cp_s = np.zeros(np.shape(M))
    cp_i = np.zeros(np.shape(M))
    for ii in range(np.shape(M)[0]):
        x, y = find_max(M[ii])
        cp_s[ii] = x
        cp_i[ii] = y
    
    system.commutation_curve = [cp_s, cp_i]


#%%
def u_smooth(sys, error = 1e-6):
    """
    A FUNCTION LIKE THE CONTROL, BUT "SMOOTHENED" SO THAT A CONTINUOUS METHOD
    FOR INTEGRATION CAN BE USED.
    """
    return 0


#%%
def tau(s, sys):
    """
    A FUNCTION THAT IS MORE LIKE WHAT MARCO AND FERNANDO WERE USING FOR THE
    REFERENCE CURVES. BASICALLY, YOU GIVE IT A COORDINATE IN s, AND IT GIVES
    YOU AN i-VALUE DEPENDING ON WHERE ON THE REFERENCE CURVE s IS:
        IF TOO SMALL, GIVES i_max
        IF TOO BIG, GIVES 0
        IF IN BETWEEN, GIVES THE i-VALUE DEPENDING ON THE CURVE.
    """
    tt = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if s < sys.sbar:
        return sys.imax
    elif s >= sys.sbar and s < sys.tau._curve_sol()[0]:
        return tt(s)
    else:
        return 0


#%%
def phi(s, sys):
    """
    A FUNCTION THAT IS MORE LIKE WHAT MARCO AND FERNANDO WERE USING FOR THE
    REFERENCE CURVES. BASICALLY, YOU GIVE IT A COORDINATE IN s, AND IT GIVES
    YOU AN i-VALUE DEPENDING ON WHERE ON THE REFERENCE CURVE s IS:
        IF TOO SMALL, GIVES i_max
        IF TOO BIG, GIVES 0
        IF IN BETWEEN, GIVES THE i-VALUE DEPENDING ON THE CURVE.
    """
    pp = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if s < sys.sstar:
        return sys.imax
    elif s >= sys.sstar and s < sys.phi._curve_sol()[0]:
        return pp(s)
    else:
        return 0


#%%
def com_curve(s, sys):
    """
    A FUNCTION THAT IS MORE LIKE WHAT MARCO AND FERNANDO WERE USING FOR THE
    REFERENCE CURVES. BASICALLY, YOU GIVE IT A COORDINATE IN s, AND IT GIVES
    YOU AN i-VALUE DEPENDING ON WHERE ON THE REFERENCE CURVE s IS:
        IF TOO SMALL, GIVES i_max
        IF TOO BIG, GIVES 0
        IF IN BETWEEN, GIVES THE i-VALUE DEPENDING ON THE CURVE.
    """
    cc = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                    sys.commutation_curve[1],
                                    kind = "cubic")
    if s < sys.commutation_curve[0][-1]:
        return sys.imax
    elif s >= sys.commutation_curve[0][-1] and s < sys.commutation_curve[0][0]:
        return cc(s)
    else:
        return 0    


#%%
def plot_curves(system):
    fig, ax = plt.subplots()
    ax.set_xlim(system.sbar, 1)
    ax.set_ylim(0, system.imax * 1.1)
    
    s = list(np.linspace(0, 1))
    T = [tau(si, system) for si in s]
    P = [phi(si, system) for si in s]
    C = [com_curve(si, system) for si in s]
    
    ax.plot(s, T)
    ax.plot(s, P)
    ax.plot(s, C)
    ax.fill_between(s, T, 0, facecolor = "blue", alpha = 0.5)
    ax.fill_between(s, C, P, facecolor = "blue", alpha = 0.5)
    ax.fill_between(s, P, 1, facecolor = "red", alpha = 0.5)
    ax.fill_between(s, T, C, facecolor = "red", alpha = 0.5)
    #return fig


#%%
if __name__ == "__main__":
    print("Hello World")
