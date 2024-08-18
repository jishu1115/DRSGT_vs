import numpy as np
import matplotlib.pyplot as plt

SGTA_plot = {
    "label": "Alg.1",
    "c": "b",
    "linestyle": "solid",
}

SGD_plot = {
    "label": "DRSGD",
    "c": "r",
    "linestyle": "dashed",
}

_SGTA_plot = {
    "label": "DRSGT",
    "c": "g",
    "linestyle": "dashdot",
}

VRSGT_plot = {
    "label": "VRSGT",
    "c": "y",
    "linestyle": "dotted",
}

cons_plot = {
    "label": "$N_k = [0.95^{-k}]$",
    "c": "g",
    "linestyle": "dashdot"
}

poly_plot = {
    "label": "$N_k = [0.85^{-k}]$",
    "c": "b",
    "linestyle": "dashed"
}

exp_plot = {
    "label": "$N_k = [0.9^{-k}]$",
    "c":"r",
    "linestyle": "solid"
}



def load_data(foldname):
    res = {

        "SGTA": {
            "value": np.load(foldname + 'value_SGTA.npy'),
            #"o_error": np.load(foldname + 'opt_error_SGTA.npy'),
            #"c_error": np.load(foldname + 'consensus_error_SGTA.npy'),
            "grad": np.load(foldname + 'grad_SGTA.npy'),
            "oracle": np.load(foldname + 'oracle_SGTA.npy'),
            "plot": SGTA_plot
        },

        "SGD": {
            "value": np.load(foldname + 'value_SGD.npy'),
            #"o_error": np.load(foldname + 'opt_error_SGD.npy'),
            #"c_error": np.load(foldname + 'consensus_error_SGD.npy'),
            "grad": np.load(foldname + 'grad_SGD.npy'),
            "oracle": np.load(foldname + 'oracle_SGD.npy'),
            "plot": SGD_plot
        },

        "1_SGTA": {
            "value": np.load(foldname + 'value_cons.npy'),
            #"o_error": np.load(foldname + 'opt_error_SGD.npy'),
            #"c_error": np.load(foldname + 'consensus_error_SGD.npy'),
            "grad": np.load(foldname + 'grad_cons.npy'),
            "oracle": np.load(foldname + 'oracle_cons.npy'),
            "plot": _SGTA_plot
        },

        #"VRSGT":{
         #   "value": np.load(foldname + 'value_VRSGT.npy'),
            #"grad": np.load(foldname + 'grad_VRSGT.npy'),
          #  "oracle": np.load(foldname + 'oracle_VRSGT.npy'),
           # "plot": VRSGT_plot
        #}
    }

    #offline = np.load(foldname + 'data_offline.npy')
    #grid = np.load(foldname + 'list_T.npy')

    return res

def load_data_1(foldname):
    res = {

        "cons": {
            "value": np.load(foldname + 'func_X_hat_cons.npy'),
            "grad": np.load(foldname + 'grad_cons.npy'),
            "plot": cons_plot
        },

        "exp": {
            "value": np.load(foldname + 'func_X_hat_exp.npy'),
            "grad": np.load(foldname + 'grad_exp.npy'),
            "plot": exp_plot
        },

        "poly": {
            "value": np.load(foldname + 'func_X_hat_poly.npy'),
            "grad": np.load(foldname + 'grad_poly.npy'),
            "plot": poly_plot
        }
    }

    offline = np.load(foldname + 'data_offline.npy')
    grid = np.load(foldname + 'list_T.npy')

    return res, offline, grid

def plot_value(res:dict,grid):
    for alg in res.values():
        plt.plot(grid,alg["gap"],linewidth=1.5,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Iteration k',fontdict={'size':18})
    plt.ylabel('optimality gap',fontdict={'size':18})
    plt.yscale('log')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)

def plot_value_1(res:dict,grid):
    for alg in res.values():
        plt.plot(grid,alg["gap"][:100],linewidth=1.5,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Iteration k',fontdict={'size':18})
    plt.ylabel('$f(\hat X_k) - f(X^*)$',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)


def plot_o_error(res:dict,grid):
    for alg in res.values():
        plt.plot(grid,alg["o_error"],linewidth=2,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Iteration t',fontdict={'size':18})
    plt.ylabel('$d_s(\hat X_k, X^*)$',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)


def plot_c_error(res:dict,grid):
    for alg in res.values():
        plt.plot(grid,alg["c_error"],linewidth=2,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Iteration k',fontdict={'size':18})
    plt.ylabel('$\|X_k - \hat X_k\|$',fontdict={'size':18})
    #plt.yscale('log')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)


def plot_grad(res:dict,grid):
    for alg in res.values():
        plt.plot(grid,alg["grad"],linewidth=1.5,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Iteration k',fontdict={'size':18})
    plt.ylabel('$\|grad(\hat X_k)\|$',fontdict={'size':18})
    #plt.yscale('log')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)

def plot_grad_1(res:dict,grid):
    for alg in res.values():
        plt.plot(grid,alg["grad"][:100],linewidth=1.5,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Iteration k',fontdict={'size':18})
    plt.ylabel('$\|grad(\hat X_k)\|$',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)