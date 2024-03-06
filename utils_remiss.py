import pandas as pd


def get_median_values_users(lang):
    if lang == 'en':
        rel_feats = load_relevant_features("cvc_data/results_selected_features_names1_fake_spreaders_en3_random_en")
        rel_fake_spreaders = load_medians_file("cvc_data/results_1_fake_spreaders_en", rel_feats)
        rel_fact_checkers = load_medians_file("cvc_data/results_2_fact_checkers_eng", rel_feats)
        rel_random = load_medians_file("cvc_data/results_3_random_en", rel_feats)
    elif lang == 'es':
        rel_feats = load_relevant_features("cvc_data/results_selected_features_names4_fake_spreaders_esp6_random_es")
        rel_fake_spreaders = load_medians_file("cvc_data/results_4_fake_spreaders_esp", rel_feats)
        rel_fact_checkers = load_medians_file("cvc_data/results_5_fact_checkers_esp", rel_feats)
        rel_random = load_medians_file("cvc_data/results_6_random_es", rel_feats)
    else:  # asumiendo catalan sin que existan otros lenguajes
        rel_feats = load_relevant_features("cvc_data/results_selected_features_names6_random_es7_fake_spreaders_cat")
        rel_fake_spreaders = load_medians_file("cvc_data/results_7_fake_spreaders_cat", rel_feats)
        rel_fact_checkers = load_medians_file("cvc_data/results_5_fact_checkers_esp", rel_feats)
        rel_random = load_medians_file("cvc_data/results_6_random_es", rel_feats)
    return rel_feats, rel_fake_spreaders, rel_fact_checkers, rel_random


def load_relevant_features(relevant_features_path):
    relevant_features = {}
    with open(relevant_features_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            feature = line.replace("\n", "")
            if ":" in feature:
                relevant_features[feature] = []
                current_feat = feature
            else:
                relevant_features[current_feat].append(feature)
    return (relevant_features)


def load_medians_file(filepath, relevant_feats):
    relevant_vals = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            line = line.split("\t")
            feature = line[0]
            median = line[1].replace("\n", "")
            for feature_type in relevant_feats:
                if feature in relevant_feats[feature_type]:
                    if feature_type not in relevant_vals:
                        relevant_vals[feature_type] = {}
                    relevant_vals[feature_type][feature] = median
    return relevant_vals


def convertDictToDataframe(relevantFeaturesDict):
    # df = pd.DataFrame();
    listOfKeys = [];
    listOfValues = []
    for dictTittle, currentDict in relevantFeaturesDict.items():
        # print("\-------------------------DIC TITTLE:", dictTittle)
        for key in currentDict.keys():
            # print("	",key + ':', currentDict[key])
            listOfKeys.append(key);
            listOfValues.append(currentDict[key])
    # df[str(key)]=currentDict[key];
    #	print ("DataFrame is ", pd)
    # print ("keys", listOfKeys)
    #	print ("values", listOfValues)
    #	print("lengths", len(listOfKeys),len(listOfValues));
    tmpDict = dict(zip(listOfKeys, listOfValues))
    df = pd.DataFrame(tmpDict, index=[0]);
    # print("    /-/*-/*-/*-/-*/*-/-*/-*/-*/-*/-DATAFRAME IS",df.to_string());
    # print ("Size is ", df.info())
    return df


def get_all_values_users(lang):
    if lang == 'en':
        rel_fake_spreaders = load_medians_file_all_features("cvc_data/results_1_fake_spreaders_en")
        rel_fact_checkers = load_medians_file_all_features("cvc_data/results_2_fact_checkers_eng")
        rel_random = load_medians_file_all_features("cvc_data/results_3_random_en")
    elif lang == 'es':
        rel_fake_spreaders = load_medians_file_all_features("cvc_data/results_4_fake_spreaders_esp")
        rel_fact_checkers = load_medians_file_all_features("cvc_data/results_5_fact_checkers_esp")
        rel_random = load_medians_file_all_features("cvc_data/results_6_random_es")
    else:  # asumiendo catalan sin que existan otros lenguajes
        rel_fake_spreaders = load_medians_file_all_features("cvc_data/results_7_fake_spreaders_cat")
        rel_fact_checkers = load_medians_file_all_features("cvc_data/results_5_fact_checkers_esp")
        rel_random = load_medians_file_all_features("cvc_data/results_6_random_es")
    return rel_fake_spreaders, rel_fact_checkers, rel_random


def load_medians_file_all_features(filepath):
    relevant_vals = {}
    relevant_vals['AllValues'] = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            line = line.split("\t")
            feature = line[0]
            median = line[1].replace("\n", "")
            relevant_vals['AllValues'][feature] = median
    return relevant_vals
