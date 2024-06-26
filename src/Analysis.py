import seg.segment as seg
import pandas as pd
from sklearn import linear_model
import numpy as np
import toolbox as tb


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
        self.Session.get_exp_info()
        self.im_size = self.Session.im_size
        self.SMs = SMs

        assert type(self.SMs) is list, "SMs must be a list (even a singleton)"

        self.SM_model = list(self.SMs[0].seg_maps.keys())[0]
        self.SM_n_components = list(self.SMs[0].seg_maps[self.SM_model].keys())[0]
        self.tag = None
        self.mode = "n/a"
        self.neuron_exclusion = False
        if self.Session._neuron_exclusion:
            self.neuron_exclusion = True
            self.neuron_exclusion_parameters = self.Session.neuron_exclusion_parameters

    def __repr__(self) -> str:
        if self.mode == "n/a":
            out1 = "Choose analysis mode: pairwise | single-neuron"
            return out1
        out2 = "\n{} Analysis object for {} SegmentationMaps: \
            \n exclusion:{}, model: {}, n_components: {}, layers: {}".format(
            self.mode,
            len(self.SMs),
            str(self.neuron_exclusion),
            self.SM_model,
            self.SM_n_components,
            str(self.layers_of_interest),
        )

        return out2

    def run_model_fit(
        self,
        mode="single-neuron",
        layers_of_interest=None,
        bounding_box=(130, 130),
        spatial_average=(20, 20),
        calculate_entropy=True,
    ) -> pd.DataFrame:
        """
        Fits SegmentationMap object to neural Session object,

        Parameters:
        -------------
        mode : str
            ('single-neuron' | 'pairwise')
        layers_of_interest : list
            list of layers in SegmentationMap to fit to Session
        bounding_box : tup of ints
            a particular area within the segmented stimulus to use for fitting
        """

        if layers_of_interest is not None:
            layers_of_interest = layers_of_interest
        else:
            layers_of_interest = [0, 4, 8, 12]

        self.mode = mode
        self.layers_of_interest = layers_of_interest

        for SM in self.SMs:
            seg._reshape_model_weights(SM, layers_of_interest)
            # constructs SM.pmap attribute from SM.model_res

        kwargs = {
            "sample_ims": len(self.SMs),
            "sample_neurons": 1,
            "random": False,
            "calculate_delta_rsc": False,
            "clean": True,
            "analysis": mode,
        }

        if mode == "single-neuron":
            from analysis import single_neuron_analysis as sna

            df = self.Session.get_df(**kwargs)

            out = sna.stitch_info(
                df,
                self.SMs,
                layers_of_interest,
                im_shape=self.Session.im_size,
                bounding_box=bounding_box,
                spatial_average=spatial_average,
            )

            if calculate_entropy:
                out["entropy"] = (
                    sum(
                        [
                            out["neuron_p{}".format(i)]
                            * np.log(out["neuron_p{}".format(i)])
                            for i in range(self.SM_n_components)
                        ]
                    )
                    * -1
                )

        elif mode == "pairwise":
            pass
            # TODO: write this code in src/analysis/pairwise_analysis.py
            from analysis import pairwise_analysis as pwa

            df = self.Session.df

            out = pwa.stitch_info(
                df,
                self.SMs,
                layers_of_interest,
                im_shape=self.Session.im_size,
                bounding_box=bounding_box,
                spatial_average=spatial_average,
            )


        self.fit_df = out
        return out

    def LinearModel(
        self,
        neural_metrics=None,
        map_metrics=None,
        control=None,
        norm=True,
        type="Linear",
        reg_kwargs={"positive": False, "fit_intercept": True},
    ):
        """
        Fits a linear model from segmentation map metrics to neural metrics ie:

        y = Wx

        where y is a neural feature, x are map features

        Parameters:
        ------------
        neural_metrics : list of str
            Metric to be predicted, if list, do one regression per metric
        map_metrics : list of str
            Metrics to use in prediction
        control : str
            Non-map, non-neural metric to be used as a control
        norm : bool, Default True
            whether data should be z-scored before regression
        type : str 
            Linear | Uses a standard linear regression 
            Ridge | Uses Ridge regressino 
        """

        assert hasattr(self, "fit_df"), "Call method run_model_fit() first"

        if type == "Linear":
            reg = linear_model.LinearRegression(**reg_kwargs)
        elif type == "Ridge":
            #TODO: implement Ridge regression option 
            pass
        
        if self.mode == "single-neuron":
            all_metrics = neural_metrics + map_metrics
            for metric in all_metrics:
                assert metric in list(
                    self.fit_df.columns
                ), "{} not in fit_df columns".format(metric)

            select = ["layer"] + all_metrics + [control]

            sdf = self.fit_df.loc[:, select]

            sdf = sdf.set_index("layer")

            if norm:
                sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min())

            # Regress with control first

            control_dfs = {k: None for k in neural_metrics}

            for metric in neural_metrics:
                out = pd.concat(
                    [
                        tb.df_regress(
                            reg,
                            sdf.loc[[i]],
                            X_labels=control,
                            y_labels=metric,
                            index_tag=i,
                        )
                        for i in range(len(self.layers_of_interest))
                    ],
                    axis=0,
                    ignore_index=True,
                )

                out.rename({"index": "layer"}, axis=1)

                control_dfs[metric] = out

            ctrl_df = control_dfs[neural_metrics[0]]

            for i, metric in enumerate(neural_metrics[1:]):
                ctrl_df.insert(i + 2, "coefs_" + metric, control_dfs[metric]["coefs"])
                ctrl_df.insert(
                    i + 4,
                    "r2 | {}".format(metric),
                    control_dfs[metric]["r2 | {}".format(metric)],
                )
                ctrl_df.insert(
                    i + 6,
                    "p_{}".format(metric),
                    control_dfs[metric]["p_{}".format(metric)],
                )
            self.ctrl_df = ctrl_df.rename(
                {"index": "layer", "coefs": "coefs_" + neural_metrics[0]}, axis=1
            )

            self.ctrl_df["predictor"] = control
            self.ctrl_df["control"] = control
            self.ctrl_df["is_control"] = 1

            # Regress with map_feature + control
            result_dfs = {k: None for k in neural_metrics}
            sdf = sdf.dropna()
            X_labels = map_metrics + [control]
            for metric in neural_metrics:
                out = pd.concat(
                    [
                        tb.df_regress(
                            reg,
                            sdf.loc[[i]],
                            X_labels=X_labels,
                            y_labels=metric,
                            index_tag=i,
                        )
                        for i in range(len(self.layers_of_interest))
                    ]
                )

                result_dfs[metric] = out

            res_df = result_dfs[neural_metrics[0]]

            for i, metric in enumerate(neural_metrics[1:]):
                res_df.insert(i + 2, "coefs_" + metric, result_dfs[metric]["coefs"])
                res_df.insert(
                    i + 4,
                    "r2 | {}".format(metric),
                    result_dfs[metric]["r2 | {}".format(metric)],
                )
                res_df.insert(
                    i + 6,
                    "p_{}".format(metric),
                    result_dfs[metric]["p_{}".format(metric)],
                )

            self.res_df = res_df.rename(
                {"index": "layer", "coefs": "coefs_" + neural_metrics[0]}, axis=1
            )
            self.res_df["predictor"] = map_metrics[0]
            self.res_df["control"] = control 
            self.res_df["is_control"] = 0

            # TODO: control has been output, now output result with map metrics
