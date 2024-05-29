import uncertainty_analysis as uca
import pandas as pd

class Analysis: 

    def __init__(self, Session, SMs, tag=None):
        """
        Initializes Analysis object

        Parameters:
        -----------
        Session : src.Session object 
        SMs : list of src.SegmentationMaps object
            Each src.SegmentationMap object should have src.SegmentationMap.model_res
            attribute for one model and number of components
        tag : str
            used to identify the current Analysis object
        """

        self.Session = Session
        self.Sessions.get_exp_info()
        self.SMs = SMs
        self.SM_model = self.SMs[0].model_res.keys()[0]
        self.SM_n_components = 
        self.tag = None
        self.mode = "n/a"
        self.neuron_exclusion = False
        if self.Session._neuron_exclusion:
            self.neuron_exclusion = True
            self.neuron_exclusion_parameters = self.Session.neuron_exclusion_parameters

        assert self.Session._neuron_exclusion, "Neurons not excluded in preprocessing"
        assert len(self.SMs) == self.Session.exp_info["n_images"], "# of SegmentationMaps does not match # of images"

    def __repr__(self) -> str:
        if self.mode == "n/a":
            out1 = "Choose analysis mode: pairwise | single-neuron"
            return out1 
        out2 = "\n{} Analysis object for {} SegmentationMaps: \
            \n exclusion:{}, model: {}, n_components: {}, layers: {}".format(
            self.mode, 
            len(self.SMs), 
            str(self.neuron_exclusion)
            self.SM_model,
            self.SM_n_components,
            str(self.layers_of_interest)
        )

        return out2 

    def run(self, 
            mode="single-neuron",
            layers_of_interest = None):
        
        self.mode = mode 
        self.layers_of_interest = layers_of_interest

        for SM in self.SMs:
            uca._reshape_model_weights(SM, layers_of_interest)
            #constructs SM.pmap attribute from SM.model_res

        kwargs = {
            "sample_ims": len(self.SMs),
            "sample_neurons": 1,
            "random": False,
            "calculate_delta_rsc": False,
            "clean":True,
            "analysis":mode
        }


        if mode == "single-neuron":

            df = self.Session.get_df(**kwargs)
            for i,SM in enumerate(self.SMs):
                coords_list = df.loc[df.img_idx==i,"neuron_np_coord"].values
                for i,layer in enumerate(layers_of_interest):
                    pmap = SM.pmaps[self.SM_model][self.SM_n_components][i]

                
    

        elif mode == "pairwise":
            kwargs["calculate_delta_rsc"] = True
            df = self.Session.get_df(**kwargs))

