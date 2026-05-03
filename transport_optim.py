# transport_app.py 

import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
import base64

warnings.filterwarnings("ignore")

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

st.set_page_config(page_title="Транспортная задача — программное решение", layout="wide")
st.title("Транспортная задача — программное решение")
st.markdown("""
Задана матрица стоимостей перевозки, объёмы поставки у поставщиков и объёмы спроса у потребителей.  
Поддерживается ручной ввод, загрузка из файла (CSV/Excel). Решение использует инициализацию методом Фогеля и оптимизацию через LP (pulp).
""")

# ---------- утилитарные функции ----------
def normalize_names(costs_df):
    """
    Возвращает costs_df, supplier_names, consumer_names.
    Если в costs_df уже есть осмысленные имена использую их, иначе генерирую S1.. и D1..
    """
    costs = costs_df.copy()
    # suppliers (строки)
    if costs.index.isnull().any() or any(str(x).strip()=="" for x in costs.index):
        supplier_names = [f"S{i+1}" for i in range(len(costs.index))]
        costs.index = supplier_names
    else:
        supplier_names = list(map(str, costs.index))
        costs.index = supplier_names
    # consumers (столбцы)
    if costs.columns.isnull().any() or any(str(x).strip()=="" for x in costs.columns):
        consumer_names = [f"D{j+1}" for j in range(len(costs.columns))]
        costs.columns = consumer_names
    else:
        consumer_names = list(map(str, costs.columns))
        costs.columns = consumer_names

    return costs, supplier_names, consumer_names

def balance_problem(costs_df, supply, demand, dummy_cost=0.0):
    """
   если суммарный supply != demand, добавляем фиктивный S0 или D0 с cost=dummy_cost.
    Возвращает balanced_costs, balanced_supply, balanced_demand, added_type ('row'/'col'/None),
    supplier_names, consumer_names
    """
    costs, supplier_names, consumer_names = normalize_names(costs_df)
    supply = list(supply)
    demand = list(demand)
    s_tot = sum(supply)
    d_tot = sum(demand)

   
    if len(supply) < len(costs.index):
        supply = supply + [0.0] * (len(costs.index) - len(supply))
    if len(demand) < len(costs.columns):
        demand = demand + [0.0] * (len(costs.columns) - len(demand))

    added = None
    if abs(s_tot - d_tot) < 1e-9:
        return costs, supply, demand, None, supplier_names, consumer_names

    if s_tot > d_tot:
        diff = s_tot - d_tot
        demand.append(diff)
        costs["D0"] = [dummy_cost]*len(costs.index)
        consumer_names = list(costs.columns)
        added = 'col'
    else:
        diff = d_tot - s_tot
        supply.append(diff)
        new_row = pd.Series([dummy_cost]*len(costs.columns), index=costs.columns, name="S0")
        costs = pd.concat([costs, new_row.to_frame().T])
        supplier_names = list(costs.index)
        added = 'row'

    return costs, supply, demand, added, supplier_names, consumer_names

def vogel_initial_solution(costs_df, supply, demand):
    """
    возвращает матрицу фогеля того же размера, что costs_df.
    """
    costs = costs_df.copy().astype(float)
    supply = supply[:]  # mutable copy
    demand = demand[:]
    m, n = len(supply), len(demand)
    alloc = np.zeros((m, n))
    row_active = [True]*m
    col_active = [True]*n

    # бескон цикл
    iteration_guard = 0
    while True:
        iteration_guard += 1
        if iteration_guard > (m+n)*1000:
            break

        if all(abs(s) < 1e-9 for s in supply) and all(abs(d) < 1e-9 for d in demand):
            break

        # штрафы в ряд
        row_pen = [-1]*m
        for i in range(m):
            if row_active[i] and supply[i] > 1e-9:
                vals = [costs.iloc[i, j] for j in range(n) if col_active[j] and demand[j] > 1e-9]
                if len(vals) >= 2:
                    vals_sorted = sorted(vals)
                    row_pen[i] = vals_sorted[1] - vals_sorted[0]
                elif len(vals) == 1:
                    row_pen[i] = vals[0]
        # штрафы в столб
        col_pen = [-1]*n
        for j in range(n):
            if col_active[j] and demand[j] > 1e-9:
                vals = [costs.iloc[i, j] for i in range(m) if row_active[i] and supply[i] > 1e-9]
                if len(vals) >= 2:
                    vals_sorted = sorted(vals)
                    col_pen[j] = vals_sorted[1] - vals_sorted[0]
                elif len(vals) == 1:
                    col_pen[j] = vals[0]

        max_row_pen = max(row_pen) if row_pen else -1
        max_col_pen = max(col_pen) if col_pen else -1
        if max_row_pen == -1 and max_col_pen == -1:
            break

        if max_row_pen >= max_col_pen:
            i = row_pen.index(max_row_pen)
            # дешевейший в ряду
            candidates = [(costs.iloc[i, j], j) for j in range(n) if col_active[j] and demand[j] > 1e-9]
            if not candidates:
                row_active[i] = False
                continue
            _, j = min(candidates)
        else:
            j = col_pen.index(max_col_pen)
            candidates = [(costs.iloc[i, j], i) for i in range(m) if row_active[i] and supply[i] > 1e-9]
            if not candidates:
                col_active[j] = False
                continue
            _, i = min(candidates)

        q = min(supply[i], demand[j])
        alloc[i, j] = q
        supply[i] -= q
        demand[j] -= q
        if supply[i] <= 1e-9:
            row_active[i] = False
            supply[i] = 0.0
        if demand[j] <= 1e-9:
            col_active[j] = False
            demand[j] = 0.0

    alloc_df = pd.DataFrame(alloc, index=costs_df.index, columns=costs_df.columns)
    return alloc_df

def total_cost(alloc_df, cost_df):
    return float((alloc_df * cost_df).sum().sum())

def solve_with_pulp(costs_df, supply, demand):
    """формурую и решаю LP через pulp (если доступен)."""
    if not PULP_AVAILABLE:
        raise RuntimeError("pulp не установлен в окружении. Установите 'pip install pulp'.")
    m = len(costs_df.index)
    n = len(costs_df.columns)
    prob = pulp.LpProblem("Transportation", pulp.LpMinimize)
    x = {}
    for i_idx in range(m):
        for j_idx in range(n):
            x[(i_idx, j_idx)] = pulp.LpVariable(f"x_{i_idx}_{j_idx}", lowBound=0, cat='Continuous')
    prob += pulp.lpSum([float(costs_df.iloc[i_idx, j_idx]) * x[(i_idx, j_idx)]
                        for i_idx in range(m) for j_idx in range(n)])
    # запас ограничения
    for i_idx in range(m):
        prob += pulp.lpSum([x[(i_idx, j_idx)] for j_idx in range(n)]) == float(supply[i_idx])
    # спрос ограничения
    for j_idx in range(n):
        prob += pulp.lpSum([x[(i_idx, j_idx)] for i_idx in range(m)]) == float(demand[j_idx])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    alloc = np.zeros((m, n))
    for i_idx in range(m):
        for j_idx in range(n):
            val = pulp.value(x[(i_idx, j_idx)])
            alloc[i_idx, j_idx] = 0.0 if val is None else float(val)
    alloc_df = pd.DataFrame(alloc, index=costs_df.index, columns=costs_df.columns)
    return alloc_df, float(pulp.value(prob.objective)), pulp.LpStatus[prob.status]

# ---------- инициализация ----------
if 'opt_done' not in st.session_state:
    st.session_state.opt_done = False
    st.session_state.alloc_opt = None
    st.session_state.opt_cost = None
    st.session_state.opt_status = None

# ---------- ввод   ----------
st.sidebar.header("Входные данные")

input_mode = st.sidebar.radio("Источник данных", ("Пример", "Загрузить файл (CSV/Excel)", "Ручной ввод (таблица)"))

costs = None
supply = None
demand = None

if input_mode == "Пример":
    st.info("Используется пример (3 поставщика × 4 потребителя, изначально несбалансирована).")
    costs = pd.DataFrame([[8,6,10,9],
                          [9,7,4,2],
                          [3,4,2,5]],
                         index=["S1","S2","S3"],
                         columns=["D1","D2","D3","D4"])
    supply = [100, 120, 120]
    demand = [80, 70, 120, 60]

elif input_mode == "Загрузить файл (CSV/Excel)":
    uploaded = st.file_uploader(
        "Загрузите CSV или Excel. Таблица должна содержать матрицу цен, спрос и предложение в листах cost, demand, supply."
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded, index_col=0)
                st.write("Загруженная таблица (интерпретируем как матрицу стоимостей):")
                st.dataframe(df)
                costs = df
               
                if 'supply' in df.columns:
                    supply = df['supply'].tolist()
                    costs = df.drop(columns=['supply'])
                if 'demand' in df.index:
                    demand = df.loc['demand'].tolist()
                    costs = df.drop(index=['demand'])

            else:  # Excel
                xls = pd.ExcelFile(uploaded)
            
            # --- читаю стоимость ---
                if 'costs' in xls.sheet_names:
                    costs = pd.read_excel(xls, sheet_name='costs', index_col=0)
                else:
                    first = xls.sheet_names[0]
                    costs = pd.read_excel(xls, sheet_name=first, index_col=0)
                st.write("Матрица стоимостей (из файла):")
                st.dataframe(costs)

            # --- читаю запас ---
                if 'supply' in xls.sheet_names:
                    s_sheet = pd.read_excel(xls, sheet_name='supply')
                    print(s_sheet)
                    try:
                        # ожидаю 2 колонки: name / value
                        supply = [float(x) for x in s_sheet.iloc[:, 1].dropna().tolist()]
                    except Exception:
                        # если один столбец
                        supply = [float(x) for x in s_sheet.squeeze().dropna().tolist()]

                # --- читаю спрос ---
                if 'demand' in xls.sheet_names:
                    d_sheet = pd.read_excel(xls, sheet_name='demand')
                    print(d_sheet)
                    try:
                        # D1 / значение
                        if d_sheet.shape[1] >= 2:
                            demand = [float(x) for x in d_sheet.iloc[:,1].dropna().tolist()]
                        else:
                            demand = [float(x) for x in d_sheet.squeeze().dropna().tolist()]
                    except Exception:
                        st.error("Не удалось прочитать лист demand. Проверьте формат: D1 - значение, D2 - значение ...")
                        demand = []
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            costs = None

else:    
    # вручную
    st.warning("Создайте матрицу стоимостей вручную.")
    rows = st.sidebar.number_input("Число поставщиков (m)", min_value=1, value=3, step=1)
    cols = st.sidebar.number_input("Число потребителей (n)", min_value=1, value=4, step=1)
    default = pd.DataFrame(np.zeros((rows, cols), dtype=float),
                           index=[f"S{i+1}" for i in range(rows)],
                           columns=[f"D{j+1}" for j in range(cols)])
    costs = st.data_editor(default, num_rows="dynamic", use_container_width=True)
    st.sidebar.write("Ввод supply (через запятую, по строкам):")
    supply_text = st.sidebar.text_area("Supply", value=", ".join(["0"]*rows))
    st.sidebar.write("Ввод demand (через запятую, по столбцам):")
    demand_text = st.sidebar.text_area("Demand", value=", ".join(["0"]*cols))
    try:
        supply = [float(x.strip()) for x in supply_text.split(",") if x.strip()!='']
        demand = [float(x.strip()) for x in demand_text.split(",") if x.strip()!='']
    except Exception:
        st.sidebar.error("Не удалось распарсить supply/demand. Укажите числа через запятую.")
        supply = [0]*rows
        demand = [0]*cols

if costs is None:
    st.warning("Матрица стоимостей не задана — загрузите файл или заполните таблицу.")
    st.stop()

# ---------- балансировка ----------
st.header("Параметры задачи")

costs, supplier_names, consumer_names = normalize_names(costs)

# покажу таблицы  
st.markdown("**Матрица стоимостей**")
st.dataframe(costs)
supply_df = pd.DataFrame({"Поставщик": supplier_names[:len(supply)], "Предложение": supply})
supply_df.index = range(1, len(supply_df)+1)
st.markdown("**Предложение*")
st.dataframe(supply_df)

consumer_labels = consumer_names[:len(demand)]
demand_df = pd.DataFrame({"Потребитель": consumer_labels, "Спрос": demand})
demand_df.index = range(1, len(demand_df)+1)
st.markdown("**Спрос**")
st.dataframe(demand_df)

s_total = sum(supply)
d_total = sum(demand)

summary_df = pd.DataFrame({
    "Показатель": ["Сумма предложения", "Сумма спроса"],
    "Значение": [s_total, d_total]
})

st.markdown("**Итого:**")
st.dataframe(summary_df, hide_index=True)
#st.subheader(f"**Сумма предложения: {s_total:.2f}; сумма спроса = {d_total:.2f}**")

# балансирую
balanced_costs, balanced_supply, balanced_demand, added, b_supplier_names, b_consumer_names = balance_problem(costs, supply, demand, dummy_cost=0.0)


st.header("Балансировка")
if added is not None:
    #st.header("Балансировка")
    st.warning(f"Задача несбалансирована — добавлен фиктивный {'потребитель (D0)' if added=='col' else 'поставщик (S0)'} для балансировки.")
    st.markdown("**Сбалансированная матрица стоимостей:**")
    st.dataframe(balanced_costs)
    b_supply_labels = list(balanced_costs.index)
    b_demand_labels = list(balanced_costs.columns)
    b_supply_df = pd.DataFrame({"Поставщик": b_supply_labels, "Предложение": balanced_supply})
    # нумерую
    b_supply_df.index = range(1, len(b_supply_df)+1)
    b_demand_df = pd.DataFrame({"Потребитель": b_demand_labels, "Спрос": balanced_demand})

    b_demand_df.index = range(1, len(b_demand_df)+1)
    st.markdown("**Сбалансированное предложение:**")
    st.dataframe(b_supply_df)
    st.markdown("**Сбалансированный спрос:**")
    st.dataframe(b_demand_df)
else:
    st.success("Задача уже сбалансирована — балансировка не требуется.")

# ---------- фогель ----------
st.header("Начальное решение методом Фогеля")
st.info("Метод Фогеля даёт быстрое начальное решение с допустимыми отклонениями от оптимума, позволяя провести оптимизацию гораздо быстрее.")
alloc_vogel = vogel_initial_solution(balanced_costs, balanced_supply[:], balanced_demand[:])
cost_vogel = total_cost(alloc_vogel, balanced_costs)
st.markdown("**Матрица решения методом Фогеля**")
st.dataframe(alloc_vogel.round(2))

summary_df = pd.DataFrame({
    "Показатель": ["Сумма предложения", "Сумма спроса"],
    "Значение": [s_total, d_total]
})

st.markdown("**Итого:**")

vogel_df = pd.DataFrame({
    "Показатель": ["Стоимость решения Фогеля"],
    "Значение": [cost_vogel]
})
st.dataframe(vogel_df, hide_index=True)

# ---------- lp оптимизация ----------
st.header("Оптимизация")

st.session_state.opt_done = False
if not PULP_AVAILABLE:
    st.warning("PuLP (LP solver) не обнаружен. Для оптимизации установите pulp: pip install pulp.")
else:
    if st.button("Найти оптимальное решение"):
        with st.spinner("Решаем задачу линейного программирования..."):
            try:
                alloc_opt, opt_cost, status = solve_with_pulp(balanced_costs, balanced_supply, balanced_demand)
                st.session_state.opt_done = True
                st.session_state.alloc_opt = alloc_opt
                st.session_state.opt_cost = opt_cost
                st.session_state.opt_status = status
                #st.success(f"Оптимизировано. Лучшая стоимость: {opt_cost:.2f}")
                st.markdown("**Оптимизированная матрица распределения:**")
                st.dataframe(alloc_opt.round(2))

                st.markdown("**Итого:**")

                opt_df = pd.DataFrame({
                   "Показатель": ["Стоимость оптимизированного решения"],
                    "Значение": [opt_cost]
                })
                st.dataframe(opt_df, hide_index=True)
            except Exception as e:
                st.error(f"Ошибка при оптимизации: {e}")

#if st.session_state.opt_done and st.session_state.alloc_opt is not None:
    #st.write("Текущее оптимальное решение (сохранено в сессии):")
  #  st.dataframe(st.session_state.alloc_opt.round(2))
    #st.markdown(f"**Лучшая стоимость (сохранено): {st.session_state.opt_cost:.2f}**")

# ---------- результат ----------
st.header("Результат")

rows = [
    ("Метод Фогеля", cost_vogel),
]

if st.session_state.opt_done is True:
    rows.append(("Оптимизация", st.session_state.opt_cost))
    result = min(cost_vogel, st.session_state.opt_cost)
else:
    result = cost_vogel

rows.append(("Результат", result))
final_df = pd.DataFrame(
    rows, columns=["Метод", "Значение"]
)

st.dataframe(final_df, hide_index=True)


# эксель если оптимизация была выполнена 5 листов иначе 4
def make_excel_bytes():
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        # 1) матрица стоимостей
        costs.to_excel(writer, sheet_name="Матрица_стоимостей")
        # 2) поставщики
        supply_export = pd.DataFrame({"Поставщик": list(balanced_costs.index)[:len(balanced_supply)], "Предложение": balanced_supply})
        supply_export.to_excel(writer, sheet_name="Поставщики", index=False)
        # 3) потребители
        demand_export = pd.DataFrame({"Потребитель": list(balanced_costs.columns)[:len(balanced_demand)], "Спрос": balanced_demand})
        demand_export.to_excel(writer, sheet_name="Потребители", index=False)
        # 4) фогель
        alloc_vogel.to_excel(writer, sheet_name="Решение_Вогеля")
        df_vogel_cost = pd.DataFrame({"Описание": ["Метод Фогеля"], "Значение": [cost_vogel]})
        df_vogel_cost.to_excel(writer, sheet_name="Вогель_стоимость", index=False)
        # 5) оптимиз


        if st.session_state.opt_done and st.session_state.alloc_opt is not None:
            st.session_state.alloc_opt.to_excel(writer, sheet_name="Оптимизация")
            df_opt_cost = pd.DataFrame({"Описание": ["Оптимизация"], "Значение": [st.session_state.opt_cost]})
            df_opt_cost.to_excel(writer, sheet_name="Оптимум_стоимость", index=False)
        final_df.to_excel(writer, sheet_name="Результат", index=False)
        
    out.seek(0)
    return out.getvalue()

excel_bytes = make_excel_bytes()

def download_link(df):
    b64 = base64.b64encode(excel_bytes).decode()
    href = f'''Нажмите 
    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        download="transport_result.xlsx">
        здесь</a>, чтобы получить файл Excel с результатами.
    '''
    st.markdown(href, unsafe_allow_html=True)

download_link(excel_bytes)

#st.download_button("Скачать файл", data=excel_bytes,
 #                  file_name="transport_result.xlsx",
##                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.subheader("Пояснения")
st.write("""
- Метод Фогеля даёт хорошую стартовую точку, по сравнению с использованием других методов, или не использования таковых вообще.  
- Для точного оптимума используется библиотека PuLP, позволяющая решить транспортную задачу даже без начального решения, однако за гораздо большее время.  
- Фиктивные поставщик/потребитель обозначены как S0 / D0 и включены в сбалансированную матрицу (в случае необходимости).
