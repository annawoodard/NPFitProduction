import xml.etree.ElementTree as ET

class lheReader(object):
    ORIGINAL = 'original'

    def __init__(self,fn=None):
        self.tree = None
        self.root = None
        self.events = []
        self.lhacode_map = {}   # Maps the integer lhacode to the corresponding newcoup string name
        self.skip = False       # Flags the sample as having no re-weighted points

        if fn is not None:
            self.tree = ET.parse(fn)
            self.root = self.tree.getroot()
            self.events = self.root.findall('./event')

    def parseModelFile(self,np_param_path):
        #NOTE: This possibly should be placed somewhere else, as it doesn't directly involve the lhe file
        """Maps the newcoup lhacode to its corresponding name
        Ex: self.lhacode_map = {11:'cA',22:'cuB',23:'cuW'}
        """
        self.lhacode_map = {}
        in_block = False
        with open(np_param_path) as f:
            for l in f.readlines():
                l = l.strip()
                if l.lower() == "block newcoup":
                    # Entering newcoup block
                    in_block = True
                    continue
                if in_block:
                    if l == "":
                        # Exiting newcoup block
                        in_block = False
                        break
                    sp_l = l.split("#",1)   # Split the line into non-comment and comment sections
                    base = sp_l[0].strip()
                    tail = sp_l[1].strip() if len(sp_l) > 1 else None
                    if len(base.split()) != 2:
                        print "ERROR: Failed to parse model file!"
                        return
                    lhacode = int(base.split()[0])
                    self.lhacode_map[lhacode] = tail

    def getCouplingName(self,lhacode):
        """Attempts to map the specified lhacode to a newcoup name
        """
        if self.lhacode_map.has_key(lhacode):
            return self.lhacode_map[lhacode]
        else:
            return lhacode

    def getWeightMap(self):
        #TODO: Figure out a better name for this function...
        """Get a dictionary mapping each weight id to a set of coefficient strengths

        Ex:
        {'mg_reweight_1':
            11: 0.1,
            23: -0.3,
            24: 0.2
        }
        """
        weight_map = {}
        if self.tree is None or self.root is None:
            return weight_map

        # Map the initial point first
        slha = self.root.find('./header/slha')
        if slha is None:
            print "WARNING: No slha tag found!"
            return weight_map
        weight_map[self.ORIGINAL] = {}
        in_block = False
        for l in slha.text.strip().split('\n'):
            l = l.strip()
            if l.lower() == "block newcoup":
                # Entering newcoup block
                in_block = True
                continue
            if in_block:
                if l == "":
                    # Exiting newcoup block
                    in_block = False
                    break
                sp_l = l.split("#",1)   # Split the line into non-comment and comment sections
                base = sp_l[0].strip()
                tail = sp_l[1].strip() if len(sp_l) > 1 else None
                if len(base) > 0:
                    coeff_id,val = base.split()
                    weight_map[self.ORIGINAL][int(coeff_id)] = float(val)

        # Map the re-weighted points next
        for weight in self.root.iter('weight'):
            wgt_id = weight.attrib.get('id')
            weight_map[wgt_id] = {}
            for l in weight.text.split('\n')[:-1]:
                coeff_id,val = l.split()[3:5]
                weight_map[wgt_id][int(coeff_id)] = float(val)
        return weight_map

    def getEvent(self,i):
        """Get a specific event
        """
        if self.tree is None:
            return None
        if i >= len(self.events):
            print "ERROR: Event index out of range!"
            return None
        return self.events[i]

    def getEventWeights(self,event_num):
        """Get original weight and re-weighted values for a specific event
        Ex: {
          'original':      0.01,
          'mg_reweight_1': 0.23,
          'mg_reweight_2': 0.03,
        }
        """
        event = self.getEvent(event_num)
        if event is None:
            return None
        
        event_weights = {}
        event_weights[self.ORIGINAL] = self.getOriginalEventWeight(event_num)

        if self.skip:
            return event_weights
        elif event.find('rwgt') is None:
            print "WARNING: LHE file doesn't have re-weighted points!"
            self.skip = True
            return event_weights

        for wgt in event.find('rwgt').iter('wgt'):
            wgt_id = wgt.attrib.get('id')
            event_weights[wgt_id] = float(wgt.text.strip())
        return event_weights

    def getOriginalEventWeight(self,event_num):
        """Get the original event weight (XWGTUP) for a specific event
        """
        event = self.getEvent(event_num)
        if event is None:
            return None
        line = event.text.strip().split('\n')[0]
        wgt = float(line.split()[2])
        return wgt

    def getWeightBounds(self):
        """Returns the largest and smallest weights in the entire sample, for all weightings
        """
        init_wgt = self.getOriginalEventWeight(0)
        if init_wgt is None:
            return None,None
        lo = init_wgt
        hi = init_wgt
        for idx in range(len(self.events)):
            event_wgts = self.getEventWeights(event_num=idx)
            for wgt_id,wgt_val in event_wgts.iteritems():
                if wgt_val < lo:
                    lo = wgt_val
                if wgt_val > hi:
                    hi = wgt_val
        return lo,hi

    def getCrossSections(self):
        #TODO: Include calculated error
        """Re-calculates the cross sections by summing over event weights
        """
        xsecs = {key: 0.0 for key in self.getWeightMap().keys()}
        for idx in range(len(self.events)):
            event_wgts = self.getEventWeights(event_num=idx)
            for wgt_id,wgt_val in event_wgts.iteritems():
                xsecs[wgt_id] += wgt_val
        return xsecs

if __name__ == "__main__":
    # Example usage
    sandbox       = "reweight_v4"
    lhe_path      = "%s/processtmp/Events/run_01/unweighted_events.lhe" % (sandbox)
    np_model_path = "%s/models/HEL_UFO/restrict_no_b_mass.dat" % (sandbox)
    
    print "Parsing tree..."
    lhe_tree  = lheReader(lhe_path)
    coeff_pts = lhe_tree.getWeightMap()
    n_events  = len(lhe_tree.events)

    lhe_tree.parseModelFile(np_model_path)

    print "Getting bounds..."
    low,high = lhe_tree.getWeightBounds()
    print "Low Weight : %s" % (low)
    print "High Weight: %s" % (high)

    print "Getting cross sections..."
    xsecs = lhe_tree.getCrossSections()
    for wgt_id in sorted(xsecs.keys()):
        if coeff_pts.has_key(wgt_id):
            print "%s: %s (%s)" % (wgt_id,xsecs[wgt_id],coeff_pts[wgt_id])
        else:
            print "%s: %s" % (wgt_id,xsecs[wgt_id])