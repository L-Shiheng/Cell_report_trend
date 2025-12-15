import streamlit as st
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fpdf import FPDF
import tempfile
import os
import shutil
import gc
import numpy as np

# === 1. 基础配置 ===
st.set_page_config(page_title="趋势图报告生成器 (大字体版)", layout="wide")

# 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
possible_fonts = ['simhei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC']
found_font = None
for font in possible_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        fig_test = plt.figure(); plt.text(0.5,0.5,'测'); plt.close(fig_test)
        found_font = font; break
    except: continue

# === 2. 配色方案 ===
COLOR_THEMES = {
    "商务蓝 (Professional Blue)": ["#2C3E50", "#34495E", "#4A6FA5", "#6D8EAD", "#94B0C7"],
    "清新绿 (Nature Green)": ["#27AE60", "#2ECC71", "#58D68D", "#82E0AA", "#ABEBC6"],
    "活力橙 (Vibrant Orange)": ["#D35400", "#E67E22", "#F39C12", "#F5B041", "#F8C471"],
    "莫兰迪 (Morandi)": ["#778899", "#8FBC8F", "#BC8F8F", "#B0C4DE", "#D8BFD8"],
    "经典柔和 (Set2 - 推荐)": sns.color_palette("Set2").as_hex(),
    "强对比 (Paired - 区分度高)": sns.color_palette("Paired").as_hex(),
    "明亮多彩 (Bright)": sns.color_palette("bright").as_hex(),
    "全光谱 (Spectral)": sns.color_palette("Spectral", n_colors=10).as_hex(),
    "深度 (Deep)": sns.color_palette("deep").as_hex(),
    "标准十色 (Tab10)": sns.color_palette("tab10").as_hex(),
}

# === 3. 核心绘图函数 ===

def create_trend_image(subset, comp_name, col_time, col_value, col_group, temp_dir, index, style_params):
    """
    生成单个化合物的【折线图+透视表】，包含防遮挡逻辑
    """
    colors = style_params['colors']
    show_legend = style_params['show_legend']
    line_width = style_params['line_width']
    
    # === 修改点 1: 再次拉高表格区域，给大字体腾地方 ===
    # height_ratios: [3, 1.6] -> 图表占3份，表格占1.6份 (之前是1.3)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1.6]})
    
    # === 1. 绘制折线图 ===
    unique_groups = subset[col_group].unique()
    current_palette = colors * (len(unique_groups) // len(colors) + 1)
    
    sns.lineplot(
        data=subset, 
        x=col_time, 
        y=col_value, 
        hue=col_group, 
        marker='o', 
        markersize=6,
        linewidth=line_width,
        palette=current_palette[:len(unique_groups)], 
        ax=ax1,
        legend=show_legend
    )
    
    # 防遮挡逻辑
    x_min = subset[col_time].min()
    x_max = subset[col_time].max()
    x_range = x_max - x_min
    if x_range == 0: x_range = 1
    
    ax1.set_xlim(left=x_min - x_range * 0.05, right=x_max + x_range * 0.35)
    
    unique_days = sorted(subset[col_time].unique())
    ax1.set_xticks(unique_days)
    
    # 细节修饰
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#DDDDDD')
    ax1.spines['bottom'].set_color('#666666')
    ax1.grid(True, linestyle='--', alpha=0.4, color='gray')
    
    ax1.set_title(str(comp_name), fontsize=14, fontweight='bold', pad=10, color='#333333')
    ax1.set_xlabel("天数 (Day)", fontsize=9)
    ax1.set_ylabel("峰面积 (Area)", fontsize=9)
    
    if show_legend:
        ax1.legend(
            fontsize=7, 
            title_fontsize=8, 
            loc='center right', 
            bbox_to_anchor=(1.0, 0.5), 
            frameon=False,
            title="Group"
        )

    # === 2. 绘制透视表 ===
    ax2.axis('off')
    
    try:
        pivot_df = subset.pivot_table(
            index=col_time, 
            columns=col_group, 
            values=col_value, 
            aggfunc='sum'
        ).fillna(0)
        
        # 数值格式化 (千分位)
        cell_text = []
        for i in range(len(pivot_df)):
            row_text = []
            for val in pivot_df.iloc[i]:
                if val % 1 == 0: 
                    s = f"{int(val):,}" 
                elif val > 1000:
                    s = f"{val:,.0f}"
                else: 
                    s = f"{val:.2f}"
                row_text.append(s)
            cell_text.append(row_text)
            
        row_labels = [str(x) for x in pivot_df.index]
        col_labels = [str(x) for x in pivot_df.columns]
        
        the_table = ax2.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # === 修改点 2: 字体暴力加大 ===
        num_cols = len(col_labels)
        # 字体策略：
        # - 4列以内: 12 (很大)
        # - 6列以内: 10 (标准正文大小)
        # - 更多列: 8 (最小保证看清)
        # (之前是 10/8/6，现在整体提升2号)
        font_size = 12 if num_cols < 4 else (10 if num_cols < 6 else 8)
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(font_size)
        
        # === 修改点 3: 进一步拉伸行高 ===
        # scale(1, 1.5): 宽度不变，高度拉伸 1.5 倍
        the_table.scale(1, 1.5) 
        
        for (r, c), cell in the_table.get_celld().items():
            if r == 0: cell.set_facecolor('#F4F6F7')
            cell.set_edgecolor('#DDDDDD')
            
    except Exception as e:
        ax2.text(0.5, 0.5, "Table Error", ha='center')

    plt.tight_layout()
    
    img_path = os.path.join(temp_dir, f"trend_{index}.png")
    plt.savefig(img_path, dpi=100, bbox_inches='tight')
    plt.close('all')
    return img_path

def generate_grid_pdf(df, col_compound, col_time, col_value, col_group, cols_per_row, style_params):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Trend Analysis Report', 0, 1, 'C')
    
    temp_dir = tempfile.mkdtemp()
    
    # A4 排版参数
    page_width = 210
    margin = 10
    usable_width = page_width - (2 * margin)
    gap = 5 
    img_width = (usable_width - (cols_per_row - 1) * gap) / cols_per_row
    img_height = img_width # 正方形
    
    x_start, y_start = margin, 25
    current_x, current_y = x_start, y_start
    page_break_y = 280 

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    df = df.dropna(subset=[col_compound])
    df = df[~df[col_compound].astype(str).str.contains("总计|Total", case=False, na=False)]
    compounds = df[col_compound].unique()
    total_items = len(compounds)
    
    try:
        for i, comp_name in enumerate(compounds):
            status_text.text(f"Processing {i+1}/{total_items}...")
            
            subset = df[df[col_compound] == comp_name]
            if len(subset) < 1: continue
            
            img_path = create_trend_image(subset, comp_name, col_time, col_value, col_group, temp_dir, i, style_params)
            
            if current_y + img_height > page_break_y:
                pdf.add_page()
                current_x = x_start
                current_y = 15
            
            pdf.image(img_path, x=current_x, y=current_y, w=img_width, h=img_height)
            
            if (i + 1) % cols_per_row == 0:
                current_x = x_start
                current_y += img_height + gap
            else:
                current_x += img_width + gap
            
            progress_bar.progress((i + 1) / total_items)
            if i % 20 == 0: gc.collect()

        out_path = os.path.join(temp_dir, "Trend_Report_Grid.pdf")
        pdf.output(out_path)
        with open(out_path, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes

    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        plt.close('all')

# === 4. Streamlit 界面 ===
st.title("📈 趋势图报告 (大字体版)")

# --- 侧边栏 ---
st.sidebar.header("🎨 样式设置")
selected_theme_name = st.sidebar.selectbox("1. 配色方案", list(COLOR_THEMES.keys()), index=5)
selected_colors = COLOR_THEMES[selected_theme_name]

line_width = st.sidebar.slider("2. 线条粗细", 1.0, 4.0, 2.0, 0.5)
show_legend = st.sidebar.checkbox("3. 显示图例", value=True)

if show_legend:
    st.sidebar.caption("✅ 已开启防遮挡：X轴将自动延长，为图例腾出空白区域。")

style_params = {
    'colors': selected_colors,
    'line_width': line_width,
    'show_legend': show_legend
}

# --- 主界面 ---
if found_font:
    st.caption(f"✅ 中文支持: {found_font}")
else:
    st.warning("⚠️ 未检测到中文字体")

uploaded_file = st.file_uploader("上传 Excel 文件 (原始数据)", type=["xlsx"])

if uploaded_file is not None:
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names
    default_idx = 0
    for idx, name in enumerate(sheet_names):
        if "表" in name or "Sheet1" in name: default_idx = idx; break
    target_sheet = st.selectbox("选择数据 Sheet:", sheet_names, index=default_idx)
    
    df = pd.read_excel(uploaded_file, sheet_name=target_sheet)
    cols = df.columns.tolist()
    
    st.write("### 字段映射")
    c1, c2, c3, c4 = st.columns(4)
    
    def get_index(options, keyword):
        for i, opt in enumerate(options):
            if keyword in str(opt): return i
        return 0

    with c1: col_compound = st.selectbox("化合物列", cols, index=get_index(cols, "化合物"))
    with c2: col_time = st.selectbox("时间列 (X轴)", cols, index=get_index(cols, "天数"))
    with c3: col_value = st.selectbox("数值列 (Y轴)", cols, index=get_index(cols, "峰面积"))
    with c4: col_group = st.selectbox("分组列 (颜色)", cols, index=get_index(cols, "培养基"))

    st.write("---")
    
    layout_col1, layout_col2 = st.columns([1, 4])
    with layout_col1:
        st.write("#### 排版设置")
        cols_per_row = st.radio("一行几个?", [1, 2, 3, 4], index=1)

    with layout_col2:
        st.write(f"#### 👁️ 效果预览 (前 {cols_per_row} 个)")
        if st.button("点击预览 (检查字体大小)"):
            temp_preview_dir = tempfile.mkdtemp()
            try:
                compounds = df[col_compound].unique()[:cols_per_row]
                if len(compounds) > 0:
                    preview_cols = st.columns(cols_per_row)
                    for i, comp_name in enumerate(compounds):
                        subset = df[df[col_compound] == comp_name]
                        p_path = create_trend_image(subset, comp_name, col_time, col_value, col_group, temp_preview_dir, i, style_params)
                        with preview_cols[i]:
                            st.image(p_path, caption=f"{comp_name}", use_column_width=True)
                    st.success("预览完成：现在的表格数字应该非常清晰了。")
            finally:
                shutil.rmtree(temp_preview_dir)

    st.write("---")
    if st.button("🚀 生成并下载 PDF"):
        with st.spinner("正在生成..."):
            pdf_bytes = generate_grid_pdf(df, col_compound, col_time, col_value, col_group, cols_per_row, style_params)
            if pdf_bytes:
                st.success("PDF 生成成功！")
                st.download_button(
                    label="📥 下载 PDF 报告",
                    data=pdf_bytes,
                    file_name=f"Trend_Report_{cols_per_row}x_LargeFont.pdf",
                    mime="application/pdf"
                )

