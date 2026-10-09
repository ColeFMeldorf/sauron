
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

scone_cut_thresh = 0.0


def get_scone_col(df):
    potentials = []
    for c in df.columns:
        if "SCONE" in c:
            potentials.append(c)

    if len(potentials) == 1:
        return potentials[0]
    else:
        raise ValueError(f"something went wrong. found {potentials}")


def scone_cut(df, scone_col = None, scone_cut_thresh = None):
    print(f"Applying scone cut with threshold {scone_cut_thresh}")
    if scone_col is None:
        scone_col = get_scone_col(df)
    return df[df[scone_col] > scone_cut_thresh]


def scone_plot(df_sim_all, df_data, survey = "", df_sim_ia = None, df_sim_cc = None):
    bins = np.linspace(scone_cut_thresh, 1, 20)
    plt.hist(scone_cut(df_sim_all, scone_cut_thresh=scone_cut_thresh)[get_scone_col(df_sim_all)], label = "All Sim",
        bins = bins, histtype = "step", density = True)
    plt.hist(scone_cut(df_data, scone_cut_thresh=scone_cut_thresh)[get_scone_col(df_data)], label = "All Real Data",
        bins = bins, histtype = "step", density = True)
    if df_sim_ia is not None:
        plt.hist(df_sim_ia[get_scone_col(df_sim_ia)], label = "Sim Ia Data", bins = bins, histtype = "step")
    if df_sim_cc is not None:
        plt.hist(df_sim_cc[get_scone_col(df_sim_cc)], label = "Sim CC Data", bins = bins, histtype = "step")
    plt.legend()
    plt.yscale("log")
    plt.title(f"SCONE score Sim and Data {survey} with PIa > {scone_cut_thresh}")


def SNRMAX_plot(df_sim_all, df_data, survey = "", column = 0):
    # plt.suptitle(survey)
    for i, snrmaxindex in zip(np.array([1, 3, 5]) + column, np.array([1, 2, 3])):
        plt.subplot(3, 2, i)
        print(f"subplot number 6, column {column + 1}, row {column * 3 + i+1}")
        plt.title(f"{survey} SNRMAX{snrmaxindex}, PIa > {scone_cut_thresh}")

        bins = np.linspace(0, 100, 20)
        plt.hist(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)[f"SNRMAX{snrmaxindex}"], density = True,
            alpha=0.5, label="Sim All", bins = bins)
        plt.hist(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)[f"SNRMAX{snrmaxindex}"], density = True,
            alpha=0.5, label="Data All", bins = bins)
        plt.legend()
        plt.yscale("log")
    # plt.tight_layout()


def mb_plot(df_sim_all, df_data, survey = "", column = 0):
    plt.subplot(1, 2, column + 1)
    plt.title(f"{survey}PIa > {scone_cut_thresh}")

    plt.hist(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)["mB"], density = True, alpha=0.5, label="Sim All")
    plt.hist(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)["mB"], density = True, alpha=0.5, label="Data All")
    plt.legend()
    plt.yscale("log")


def mb_err_plot(df_sim_all, df_data, survey = "", column = 0):
    plt.subplot(1, 2, column + 1)
    plt.title(f"{survey}PIa > {scone_cut_thresh}")

    plt.hist(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)["mBERR"], density = True, alpha=0.5, label="Sim All")
    plt.hist(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)["mBERR"], density = True, alpha=0.5, label="Data All")
    plt.legend()
    plt.yscale("log")


def sim_genz_plot(df_sim_cc, df_sim_ia, column = 0, survey = "", ):
    plt.subplot(1, 2, column + 1)
    plt.title(f"{survey} Sim Gen Z")

    if "scone" in survey.lower():
        histtype = "step"
    else:
        histtype = "bar"

    try:
        plt.hist(df_sim_cc["SIM_ZCMB"], alpha=0.5, label="Sim CC", histtype=histtype)
        plt.hist(df_sim_ia["SIM_ZCMB"], alpha=0.5, label="Sim IA", histtype=histtype)
        plt.legend()
        plt.yscale("log")
    except:
        plt.hist(df_sim_cc["GENZ"], alpha=0.5, label="Sim CC", histtype=histtype)
        plt.hist(df_sim_ia["GENZ"], alpha=0.5, label="Sim IA", histtype=histtype)
        plt.legend()
        plt.yscale("log")



# sdss_base_data_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/CFM-SDSS-JH8/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"
# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-SMALL/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-SMALL_SDSS-0001/FITOPT001.FITRES.gz"
sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-SMALL-NEWSCONE/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-SMALL-NEWSCONE_SDSS-0001/FITOPT000.FITRES.gz"
sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-SMALL-NEWSCONE/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"
sdss_df_sim_all = pd.read_csv(sdss_base_path_sim_all, sep = r"\s+", comment="#")
sdss_df_sim_ia = sdss_df_sim_all[np.isin(sdss_df_sim_all["SIM_GENTYPE"], [1])]
sdss_df_sim_cc = sdss_df_sim_all[~np.isin(sdss_df_sim_all["SIM_GENTYPE"], [1])]
sdss_df_data = pd.read_csv(sdss_base_data_path, sep = r"\s+", comment="#")
sdss_dump_all_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/CFM-SDSS-JH8/1_SIM/SDSS_BCOR/PIP_CFM-SDSS-JH8_SDSS_BCOR/PIP_CFM-SDSS-JH8_SDSS_BCOR.DUMP.gz"
columns = pd.read_csv(sdss_dump_all_path, sep = r"\s+", comment="#",  nrows=0).columns.tolist()

sdss_dump_all = pd.read_csv(sdss_dump_all_path, sep = r"\s+", comment="#", usecols = ["GENZ", "GENTYPE"])
sdss_dump_ia = sdss_dump_all[np.isin(sdss_dump_all["GENTYPE"], [1])]
sdss_dump_cc = sdss_dump_all[~np.isin(sdss_dump_all["GENTYPE"], [1])]
# for c in sdss_df_sim_all.columns:
#     print(c)
# import pdb; pdb.set_trace()

des_base_data_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/5_MERGE/MERGE_Dsys_DATADESSMP/output/DES-SN5YR_DES/FITOPT000.FITRES.gz"
des_base_path_sim = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/5_MERGE/MERGE_Dsys_DATADES_SIM_IA/output/PIP_D5YR_RATEPZ_NEWCC_SIM_NOMINAL_DATADESSIM_IA-0001/FITOPT000.FITRES.gz"
des_base_path_sim_cc = des_base_path_sim.replace("SIM_IA", "SIM_CC")
des_df_sim_ia = pd.read_csv(des_base_path_sim, sep = r"\s+", comment="#")
des_df_sim_cc = pd.read_csv(des_base_path_sim_cc, sep = r"\s+", comment="#")
des_df_data = pd.read_csv(des_base_data_path, sep = r"\s+", comment="#")
des_df_sim_all = pd.concat([des_df_sim_ia, des_df_sim_cc])
des_df_sim_all["PROB_SCONE"] = pd.concat([des_df_sim_ia[get_scone_col(des_df_sim_ia)], des_df_sim_cc[get_scone_col(des_df_sim_cc)]])
# delete other prob scone columns
des_df_sim_all = des_df_sim_all.drop(columns=[c for c in des_df_sim_all.columns if "SCONE" in c and c != "PROB_SCONE"])


des_dump_cc_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/1_SIM/DESCC_NOMINAL/PIP_D5YR_RATEPZ_NEWCC_BCOR_NOMINAL_DESCC_NOMINAL/PIP_D5YR_RATEPZ_NEWCC_BCOR_NOMINAL_DESCC_NOMINAL.DUMP.gz"
des_dump_cc = pd.read_csv(des_dump_cc_path, sep = r"\s+", comment="#", usecols = ["GENZ"])
des_dump_ia_path = des_dump_cc_path.replace("DESCC", "DESIA")
des_dump_ia = pd.read_csv(des_dump_ia_path, sep = r"\s+", comment="#", usecols = ["GENZ"])

all_dfs_des = [des_df_sim_all, des_df_sim_ia, des_df_sim_cc, des_df_data]
all_dfs_sdss = [sdss_df_sim_all, sdss_df_sim_ia, sdss_df_sim_cc, sdss_df_data]
all_dfs = all_dfs_des + all_dfs_sdss


zcol = "zHD"
for i, df in enumerate(all_dfs_des):
    all_dfs_des[i] = df[df[zcol] < 0.9]
    all_dfs_des[i] = all_dfs_des[i][all_dfs_des[i][zcol] > 0.1]

for i, df in enumerate(all_dfs_sdss):
    all_dfs_sdss[i] = df[df[zcol] < 0.35]
    all_dfs_sdss[i] = all_dfs_sdss[i][all_dfs_sdss[i][zcol] > 0.05]

plt.figure(figsize=(12, 6), dpi = 200)
plt.subplot(1, 2, 1)
scone_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", df_sim_ia = all_dfs_des[1], df_sim_cc = all_dfs_des[2])
plt.subplot(1, 2, 2)
scone_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", df_sim_ia = all_dfs_sdss[1], df_sim_cc = all_dfs_sdss[2])
plt.savefig("plotting_scripts/scone_comparison.png")
plt.close()
print("Scone comparison plot saved as 'scone_comparison.png'")


plt.figure(figsize = (12, 6), dpi = 200)
SNRMAX_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
SNRMAX_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/SNRMAX_comparison.png")
plt.close()
print("SNRMAX comparison plots saved as 'SNRMAX_comparison.png'")


plt.figure(figsize = (12, 6), dpi = 200)
mb_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
mb_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/mB_comparison.png")
plt.close()
print("mB comparison plots saved as 'mB_comparison.png'")


plt.figure(figsize = (12, 6), dpi = 200)
mb_err_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
mb_err_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/mBERR_comparison.png")
plt.close()
print("mBERR comparison plots saved as 'mBERR_comparison.png'")


plt.figure(figsize = (12, 6), dpi = 200)
sim_genz_plot(all_dfs_des[2], all_dfs_des[1], column = 0, survey = "DES")
sim_genz_plot(all_dfs_sdss[2], all_dfs_sdss[1], column = 1, survey = "SDSS")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/dump_genz_comparison.png")
plt.close()
print("Sim GENZ comparison plots saved as 'dump_genz_comparison.png'")



scone_SDSS_df_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-RATE/1_SIM/SDSS_BCOR/PIP_CFM-SDSS-RATE_SDSS_BCOR/PIP_CFM-SDSS-RATE_SDSS_BCOR.DUMP.gz"
scone_SDSS_df = pd.read_csv(scone_SDSS_df_path, comment = "#", sep=r"\s+", usecols=["GENZ", "GENTYPE"])

scone_SDSS_df = scone_SDSS_df[scone_SDSS_df["GENZ"] > 0.05]
scone_SDSS_df = scone_SDSS_df[scone_SDSS_df["GENZ"] < 0.35]
scone_SDSS_df_cc = scone_SDSS_df[~np.isin(scone_SDSS_df["GENTYPE"], [1])]
scone_SDSS_df_ia = scone_SDSS_df[np.isin(scone_SDSS_df["GENTYPE"], [1])]

des_dump_cc = des_dump_cc[des_dump_cc["GENZ"] > 0.1]
des_dump_cc = des_dump_cc[des_dump_cc["GENZ"] < 0.9]
des_dump_ia = des_dump_ia[des_dump_ia["GENZ"] > 0.1]
des_dump_ia = des_dump_ia[des_dump_ia["GENZ"] < 0.9]

sdss_dump_cc = sdss_dump_cc[sdss_dump_cc["GENZ"] > 0.05]
sdss_dump_cc = sdss_dump_cc[sdss_dump_cc["GENZ"] < 0.35]
sdss_dump_ia = sdss_dump_ia[sdss_dump_ia["GENZ"] > 0.05]
sdss_dump_ia = sdss_dump_ia[sdss_dump_ia["GENZ"] < 0.35]



plt.figure(figsize = (12, 6), dpi = 200)
sim_genz_plot(des_dump_cc, des_dump_ia, column = 0, survey = "DES")
sim_genz_plot(sdss_dump_cc, sdss_dump_ia, column = 1, survey = "SDSS")
print("Total size sdss_dump_cc:", sdss_dump_cc.shape)
print("Total size sdss_dump_ia:", sdss_dump_ia.shape)
print("Total size scone_SDSS_df_cc:", scone_SDSS_df_cc.shape)
print("Total size scone_SDSS_df_ia:", scone_SDSS_df_ia.shape)
sim_genz_plot(scone_SDSS_df_cc, scone_SDSS_df_ia, column = 1, survey = "SDSS scone trainset")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/sim_genz_comparison.png")
plt.close()
print("Sim GENZ comparison plots saved as 'sim_genz_comparison.png'")
