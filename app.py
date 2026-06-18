import numpy as np
import pandas as pd
import xgboost as xgb
import gradio as gr

# 1. Dataset Generation (1500 Samples)
np.random.seed(101)
n_samples = 1500

df = pd.DataFrame({
    'Core_Material': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.3, 0.2, 0.2]),
    'Particle_Shape': np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2]),
    'Hydrodynamic_Diameter': np.random.uniform(10, 220, n_samples),
    'PDI': np.random.uniform(0.05, 0.45, n_samples),
    'Zeta_Potential': np.random.uniform(-45, 45, n_samples),
    'LogP': np.random.uniform(-0.5, 5.5, n_samples),
    'PSA': np.random.uniform(15, 160, n_samples),
    'pKa': np.random.uniform(3.5, 10.5, n_samples),
    'Plasma_Protein_Binding': np.random.uniform(5, 98, n_samples),
    'Targeting_Ligand': np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
})

pH = 7.4
df['Fraction_Unionized'] = 1 / (1 + 10**(df['pKa'] - pH))
df['Papp_Target'] = (df['LogP'] * 2.1) - (df['PSA'] * 0.03) + (df['Fraction_Unionized'] * 3.5) - (df['Hydrodynamic_Diameter'] * 0.02)
df.loc[df['Targeting_Ligand'] > 0, 'Papp_Target'] += 8.5
df.loc[df['Particle_Shape'] == 1, 'Papp_Target'] += 1.2

df['Cytotoxicity_Target'] = (df['Zeta_Potential'].abs() * 0.8) + (df['Hydrodynamic_Diameter'] * 0.05)
df.loc[df['Core_Material'] == 3, 'Cytotoxicity_Target'] += 15.0
df.loc[df['PDI'] > 0.3, 'Cytotoxicity_Target'] += 10.0
df['Cytotoxicity_Target'] = np.clip(df['Cytotoxicity_Target'], 5, 95)

features_list = ['Core_Material', 'Particle_Shape', 'Hydrodynamic_Diameter', 'PDI', 'Zeta_Potential', 'LogP', 'PSA', 'pKa', 'Plasma_Protein_Binding', 'Targeting_Ligand']
X = df[features_list]

papp_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=101).fit(X, df['Papp_Target'])
tox_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=101).fit(X, df['Cytotoxicity_Target'])

# 2. Prediction Function
def industrial_screening(core_name, shape_name, diameter, pdi, zeta, pka, ppb, ligand_name):
    try:
        core_dict = {"Liposome / Nanoparticle Lipid Carrier (NLC)": 0, "Polymeric Nanoparticle (PLGA/PEG)": 1, "Mesoporous Silica Nanoparticle": 2, "Metallic Core (Gold / Iron Oxide)": 3}
        shape_dict = {"Spherical (Highest Stability)": 0, "Rod / Tubular": 1, "Disk-shaped": 2}
        ligand_dict = {"None (Passive Diffusion Only)": 0, "Transferrin Receptor Ligand (Active Targeted)": 1, "Angiopep-2 (LRP1 Mediated Gateway)": 2}
        
        c_val = int(core_dict.get(core_name, 0))
        s_val = int(shape_dict.get(shape_name, 0))
        l_val = int(ligand_dict.get(ligand_name, 0))
        
        input_data = pd.DataFrame([[
            c_val, s_val, float(diameter), float(pdi), float(zeta), float(pka), float(ppb), l_val
        ]], columns=features_list)
        
        pred_papp = float(papp_model.predict(input_data)[0])
        pred_cytotox = float(tox_model.predict(input_data)[0])
        frac_unionized = 1 / (1 + 10**(float(pka) - 7.4))
        
        if pred_papp > 6.0 and pred_cytotox < 35:
            decision = "SUCCESS: Optimal Formulation Approved for Pre-clinical Phase"
        elif pred_papp > 6.0 and pred_cytotox >= 35:
            decision = "WARNING: Target Reached but Exceeds Toxicity Limits"
        else:
            decision = "CRITICAL: Insufficient Permeability Profile"

        report_md = f"""
        ### NanoMS-Predict: Industrial R&D Metrics
        * **Apparent Permeability (Papp):** `{pred_papp:.2f} × 10⁻⁶ cm/s`
        * **Predicted Neuro-Cytotoxicity:** `{pred_cytotox:.1f}%`
        * **Fraction Un-ionized (at Blood pH 7.4):** `{frac_unionized*100:.1f}%`
        * **Batch Homogeneity (PDI Score):** `{pdi:.2f}`
        """
        return decision, report_md
    except Exception as e:
        return "Processing Error", str(e)

# 3. Gradio Interface Layout
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# NanoMS-Predict: Enterprise R&D Platform")
    
    with gr.Row():
        with gr.Column():
            core_name = gr.Dropdown(choices=["Liposome / Nanoparticle Lipid Carrier (NLC)", "Polymeric Nanoparticle (PLGA/PEG)", "Mesoporous Silica Nanoparticle", "Metallic Core (Gold / Iron Oxide)"], value="Liposome / Nanoparticle Lipid Carrier (NLC)", label="Nanoparticle Core Matrix")
            shape_name = gr.Dropdown(choices=["Spherical (Highest Stability)", "Rod / Tubular", "Disk-shaped"], value="Spherical (Highest Stability)", label="Geometrical Shape")
            diameter = gr.Slider(10, 220, value=90, step=1, label="Diameter (nm)")
            pdi = gr.Slider(0.05, 0.45, value=0.15, step=0.01, label="PDI (Batch Homogeneity)")
            zeta = gr.Slider(-45, 45, value=15, step=1, label="Zeta Potential (mV)")
            pka = gr.Slider(3.5, 10.5, value=7.4, step=0.1, label="pKa")
            ppb = gr.Slider(5, 98, value=65, step=1, label="Plasma Protein Binding (% PPB)")
            ligand_name = gr.Dropdown(choices=["None (Passive Diffusion Only)", "Transferrin Receptor Ligand (Active Targeted)", "Angiopep-2 (LRP1 Mediated Gateway)"], value="None (Passive Diffusion Only)", label="Targeting System / Ligand")
            btn = gr.Button("Execute Full Simulation", variant="primary")
            
        with gr.Column():
            decision_out = gr.Textbox(label="Manufacturing Decision")
            report_out = gr.Markdown("### Statistics Summary\n*Run simulation to generate analytics.*")
            
    btn.click(fn=industrial_screening, inputs=[core_name, shape_name, diameter, pdi, zeta, pka, ppb, ligand_name], outputs=[decision_out, report_out])

if __name__ == "__main__":
    demo.launch()
