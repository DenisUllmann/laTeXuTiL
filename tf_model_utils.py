# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:06:05 2023

@author: Denis
"""
import numpy as np
import tensorflow as tf

def get_sequential_state(model):
    if model.__class__.__name__ == "Sequential":
        return True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        return True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(tf.nest.flatten(v[0].keras_inputs)) > 1
            ):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break
        return sequential_like

def tf2Mod2TeX(model, modelName='unknown', modelLabel='unknown', 
               char_dbl='=', lay_line_sep='empty'):
    """
    Parameters
    ----------
    model : TYPE model with .summary() method: tf.keras.Model or keras Model
    modelName : TYPE str, optional
        DESCRIPTION. The default is 'unknown'. 
    modelLabel : TYPE str, optional
        DESCRIPTION. The default is 'unknown'.
                     The name for laTeX table label
    col_keys : TYPE list[str], optional
        DESCRIPTION. The default is ['Layer (type)', 'Output Shape', "Param #", 'Connected to'].
                     List of the header from mode.summary() method
    char_dbl : TYPE str, optional
        DESCRIPTION. The default is '='.
                     Character used by model.summary() for the double lines
    lay_line_sep : TYPE str, optional
        DESCRIPTION. The default is 'empty'.
                     Table line sep between layers
                     Choose between 'empty', None, or any laTeX readable 
                     command (eg. '\\midrule' and not '\midrule')
    
    Returns
    -------
    out_str : TYPE str
        DESCRIPTION. str for laTeX: you can save it with:
            f = open('namefile.txt', 'w')
            f.write(tf2Mod2TeX(model))
            f.close()
    """
    stringlist = []
    model.summary(line_length=150, print_fn=lambda x: stringlist.append(x))
    if get_sequential_state(model):
        col_keys = ['Layer (type)', 'Output Shape', 
                    "Param #"]
    else:
        col_keys = ['Layer (type)', 'Output Shape', 
                    "Param #", 'Connected to']
    col_format = False
    count_dbl = 0
    stringlist[0] = stringlist[0].split(':')
    stringlist[0] = '\\multicolumn{%i}{l}{%s: %s}\\\\'%(
        len(col_keys),stringlist[0][0], modelName)
    ix = 1
    tot_len = len(stringlist)
    while ix<tot_len:
        tmp = stringlist[ix]
        new_line = 0
        if all(v in tmp for v in col_keys):
            col_format = True
            col_idx = [tmp.index(v) for v in col_keys]
        if count_dbl > 1:
            col_format = False
        if all(v==char_dbl for v in tmp):
            stringlist[ix] = '\\midrule\\midrule'
            count_dbl += 1
        elif all(v=='_' for v in tmp):
            stringlist[ix] = '\\midrule'
        elif all(v==' ' for v in tmp):
            # line sep between layers
            if lay_line_sep=='empty':
                stringlist[ix] = '&'*(len(col_keys)-1)+'\\\\'
            elif lay_line_sep is None:
                tot_len -= 1
                new_line = -1
                stringlist[ix:] = stringlist[ix+1:]
            else:
                stringlist[ix] = lay_line_sep
        elif col_format:
            # this string is formatted in columns
            stringlist[ix] = [
                stringlist[ix][idx:col_idx[ic+1]
                           ] if ic<len(col_idx)-1 else stringlist[ix][
                               idx:] for ic, idx in enumerate(col_idx)]
            stringlist[ix] = [v.split(' ') for v in stringlist[ix]]
            stringlist[ix] = [v[np.argmin([vv=='' for vv in v]):
                                ] if v[0]=='' else v for v in stringlist[ix]]
            stringlist[ix] = [v[:-np.argmin([vv=='' for vv in v][::-1])
                                ] if v[-1]=='' else v for v in stringlist[ix]]
            stringlist[ix] = [' '.join(v) for v in stringlist[ix]]
            if all(v in stringlist[ix][0] for v in ['(',')']):
                # first cell is 'Name (Class)' format and needs 2 vertical TeX cells (multirow for others)
                p_idx = [stringlist[ix][0].index(v) for v in ['(',')']]
                assert p_idx[1]==len(stringlist[ix][0])-1, "modify code here, only 'nameLayer (classLayer)' format was considered"
                stringlist[ix+2:] = stringlist[ix+1:]
                stringlist[ix+1] = [stringlist[ix][0][p_idx[0]:p_idx[1]+1]]+[
                                    '']*(len(col_keys)-1)
                stringlist[ix+1] = '&'.join(stringlist[ix+1])+'\\\\'
                stringlist[ix] = [stringlist[ix][0][0:p_idx[0]]]+[
                    '\\multirow{2}{*}{%s}'%(stringlist[ix][idx_sl+1]
                                            ) for idx_sl in range(
                                                len(stringlist[ix])-1)]
                tot_len += 1
                new_line = 1
            elif stringlist[ix][0]=='' and any(
                    v!='' for v in stringlist[ix][1:]):
                # first cell is empty and at least not one of the others: 
                # this line should be merged with ix-1 and ix-2
                assert all(stringlist[ixm12][-2:]=='\\\\' for ixm12 in [ix-1,ix-2]), "wrong format, '\\\\' expected at the end of the line"
                ixm2 = stringlist[ix-2][:-2].split('&')
                ixm1 = stringlist[ix-1][:-2].split('&')
                assert ixm1[0][0]=='(' and ixm1[0][-1]==')', "modify code here, only 'nameLayer (classLayer)' format was considered for the 2 rows above"
                for iv, v in enumerate(stringlist[ix]):
                    if v!='':
                        ixm1[iv] = v
                        if 'multirow' in ixm2[iv]:
                            ixm2[iv] = ixm2[iv].split('}')[-2].split('{')[-1]
                stringlist[ix-2] = '&'.join(ixm2)+'\\\\'
                stringlist[ix-1] = '&'.join(ixm1)+'\\\\'
                tot_len -= 1
                new_line = -1
            stringlist[ix] = '&'.join(stringlist[ix])+'\\\\'
            if new_line==-1:
                stringlist[ix:] = stringlist[ix+1:]
        else:
            # this string is not formatted in columns
            stringlist[ix] = stringlist[ix].split(' ')
            stringlist[ix] = stringlist[ix][
                np.argmin([v=='' for v in stringlist[ix]]):
                ] if stringlist[ix][0]=='' else stringlist[ix]
            stringlist[ix] = stringlist[ix][
                :-np.argmin([v=='' for v in stringlist[ix]][::-1])
                ] if stringlist[ix][-1]=='' else stringlist[ix]
            stringlist[ix] = '\\multicolumn{%i}{l}{%s}\\\\'%(
                len(col_keys), ' '.join(stringlist[ix]))
        ix += 1+new_line
    prefix = ["\\begin{table}[]", "\\begin{tabular}{%s}"%('l'*len(col_keys))]
    suffix = ["\end{tabular}", "\caption{{Model summary of %s.}}"%modelName, "\label{tab:%s-model-summary}"%modelLabel , "\end{table}"]
    stringlist = prefix + ['\\toprule'] + stringlist[:-1] + ['\\bottomrule'] + suffix 
    out_str = "\n".join(stringlist)
    out_str = out_str.replace("_", "\_")
    out_str = out_str.replace("#", "\#")
    return out_str