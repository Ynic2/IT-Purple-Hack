import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image
from customtkinter import CTkButton, CTkFrame, CTkRadioButton, CTkLabel, CTkEntry, CTkImage, CTk, CTkToplevel
from CTkMessagebox import CTkMessagebox

import pandas as pd
import pyarrow.parquet as pq
import os

import xgboost as xgb



class App(CTk):
    def __init__(self):
        super().__init__()

        self.toplevel_window = None
        self.geometry("480x300")
        self.title("Prediction")
        self.iconbitmap("sberbank_icon-icons.com_71976.ico")
        self.resizable(False, False)

        self.logo = CTkImage(dark_image=Image.open("img.png"), size=(460, 150))
        self.logo_label = CTkLabel(master=self, text="", image=self.logo)
        self.logo_label.grid(row=0, column=0)

        # выбор файла предикта
        self.password_frame = CTkFrame(master=self, fg_color="transparent")
        self.password_frame.grid(row=1, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")

        self.entry_password = CTkEntry(master=self.password_frame, width=300)
        self.entry_password.grid(row=0, column=0, padx=(0, 20))

        self.btn_generate = CTkButton(master=self.password_frame, text="Open Prediction File", width=125,
                                      command=self.open_file_dialog)
        self.btn_generate.grid(row=0, column=1)

        # выбор папки
        self.text_entry = CTkEntry(master=self.password_frame, width=300)
        self.text_entry.grid(row=1, column=0, padx=(0, 20), pady=(10, 0))

        self.btn_choose_file = CTkButton(master=self.password_frame, text="Choose Folder", width=125,
                                         command=self.open_folder_dialog)
        self.btn_choose_file.grid(row=1, column=1, pady=(10, 0))

        self.btn_predict = CTkButton(master=self.password_frame, text="Predict", width=125,
                                     command=self.predict_file)
        self.btn_predict.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        # выбор ии
        # self.radio_frame = MyRadioFrame(self.password_frame, title="Choose AI",
        #                                 values=["NEURAL NETWORK", "GRADIENT BOOSTING"],
        #                                 prediction_callback=self.predict_file)
        # self.radio_frame.grid(row=2, column=0, columnspan=2, padx=(0, 20), pady=(20, 0), sticky="nsew")
        #
        # # Кнопка предикта


    # выбрать файл
    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet")])
        if file_path:
            self.entry_password.delete(0, 'end')
            self.entry_password.insert(0, file_path)
            # You can do further processing with the selected file path

    # открыть папку
    def open_folder_dialog(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.text_entry.delete(0, 'end')
            self.text_entry.insert(0, folder_path)
            # You can do further processing with the selected folder path

    # Предсказать файл
    def predict_file(self):

        file_path = self.entry_password.get()  # Получаем путь к файлу из поля ввода
        folder_path = self.text_entry.get()

        # Проверка на ввод данных
        if not file_path or not folder_path:
            show_warning()
            return


        file_name, file_extension = os.path.splitext(file_path)

        feature_fil = ['id', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature8',
                       'feature9', 'feature12', 'feature16', 'feature17', 'feature18', 'feature19', 'feature21',
                       'feature22', 'feature24', 'feature25', 'feature30', 'feature31', 'feature32', 'feature33',
                       'feature34', 'feature35', 'feature37', 'feature38', 'feature40', 'feature41', 'feature43',
                       'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49', 'feature50',
                       'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57',
                       'feature59', 'feature62', 'feature63', 'feature64', 'feature67', 'feature71', 'feature72',
                       'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature79', 'feature81',
                       'feature83', 'feature84', 'feature86', 'feature88', 'feature89', 'feature90', 'feature91',
                       'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98', 'feature99',
                       'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105', 'feature106',
                       'feature107', 'feature108', 'feature109', 'feature110', 'feature111', 'feature112', 'feature113',
                       'feature114', 'feature115', 'feature116', 'feature117', 'feature118', 'feature119', 'feature120',
                       'feature121', 'feature122', 'feature123', 'feature124', 'feature125', 'feature126', 'feature127',
                       'feature128', 'feature129', 'feature130', 'feature132', 'feature133', 'feature134', 'feature135',
                       'feature136', 'feature137', 'feature138', 'feature139', 'feature141', 'feature142', 'feature143',
                       'feature145', 'feature147', 'feature148', 'feature149', 'feature150', 'feature151', 'feature152',
                       'feature153', 'feature154', 'feature155', 'feature156', 'feature157', 'feature158', 'feature159',
                       'feature161', 'feature162', 'feature163', 'feature164', 'feature165', 'feature166', 'feature167',
                       'feature168', 'feature169', 'feature170', 'feature171', 'feature172', 'feature173', 'feature174',
                       'feature175', 'feature176', 'feature177', 'feature178', 'feature179', 'feature180', 'feature181',
                       'feature182', 'feature183', 'feature184', 'feature185', 'feature186', 'feature187', 'feature188',
                       'feature189', 'feature190', 'feature191', 'feature192', 'feature193', 'feature194', 'feature195',
                       'feature196', 'feature197', 'feature198', 'feature199', 'feature200', 'feature201', 'feature204',
                       'feature206', 'feature207', 'feature208', 'feature209', 'feature210', 'feature212', 'feature214',
                       'feature217', 'feature218', 'feature219', 'feature220', 'feature222', 'feature226', 'feature231',
                       'feature234', 'feature238', 'feature252', 'feature253', 'feature257', 'feature260', 'feature262',
                       'feature263', 'feature264', 'feature265', 'feature268', 'feature269', 'feature270', 'feature275',
                       'feature277', 'feature280', 'feature282', 'feature283', 'feature284', 'feature286', 'feature287',
                       'feature288', 'feature290', 'feature291', 'feature296', 'feature299', 'feature300', 'feature303',
                       'feature304', 'feature305', 'feature308', 'feature309', 'feature310', 'feature313', 'feature315',
                       'feature316', 'feature317', 'feature318', 'feature319', 'feature320', 'feature322', 'feature328',
                       'feature330', 'feature331', 'feature332', 'feature334', 'feature335', 'feature336', 'feature338',
                       'feature339', 'feature340', 'feature341', 'feature342', 'feature343', 'feature344', 'feature345',
                       'feature346', 'feature347', 'feature348', 'feature349', 'feature350', 'feature351', 'feature352',
                       'feature353', 'feature354', 'feature355', 'feature356', 'feature357', 'feature358', 'feature359',
                       'feature360', 'feature361', 'feature362', 'feature366', 'feature367', 'feature368', 'feature369',
                       'feature370', 'feature371', 'feature373', 'feature374', 'feature375', 'feature376', 'feature377',
                       'feature378', 'feature379', 'feature383', 'feature384', 'feature385', 'feature386', 'feature395',
                       'feature396', 'feature398', 'feature401', 'feature402', 'feature405', 'feature409', 'feature411',
                       'feature412', 'feature414', 'feature415', 'feature416', 'feature417', 'feature421', 'feature422',
                       'feature426', 'feature427', 'feature428', 'feature429', 'feature432', 'feature433', 'feature434',
                       'feature435', 'feature436', 'feature437', 'feature438', 'feature440', 'feature441', 'feature442',
                       'feature443', 'feature444', 'feature445', 'feature446', 'feature447', 'feature448', 'feature449',
                       'feature450', 'feature451', 'feature452', 'feature453', 'feature454', 'feature455', 'feature456',
                       'feature457', 'feature458', 'feature459', 'feature460', 'feature461', 'feature462', 'feature464',
                       'feature465', 'feature467', 'feature468', 'feature469', 'feature470', 'feature472', 'feature473',
                       'feature474', 'feature475', 'feature476', 'feature477', 'feature479', 'feature482', 'feature485',
                       'feature486', 'feature487', 'feature488', 'feature489', 'feature491', 'feature493', 'feature494',
                       'feature495', 'feature497', 'feature499', 'feature500', 'feature501', 'feature502', 'feature503',
                       'feature504', 'feature505', 'feature506', 'feature507', 'feature508', 'feature509', 'feature510',
                       'feature512', 'feature513', 'feature514', 'feature515', 'feature516', 'feature517', 'feature518',
                       'feature520', 'feature523', 'feature524', 'feature525', 'feature526', 'feature527', 'feature528',
                       'feature529', 'feature530', 'feature531', 'feature532', 'feature533', 'feature534', 'feature535',
                       'feature537', 'feature538', 'feature539', 'feature540', 'feature541', 'feature542', 'feature543',
                       'feature544', 'feature545', 'feature546', 'feature547', 'feature548', 'feature550', 'feature551',
                       'feature553', 'feature554', 'feature557', 'feature559', 'feature560', 'feature561', 'feature564',
                       'feature569', 'feature572', 'feature588', 'feature591', 'feature597', 'feature603', 'feature605',
                       'feature609', 'feature615', 'feature620', 'feature624', 'feature626', 'feature632', 'feature646',
                       'feature647', 'feature651', 'feature652', 'feature653', 'feature654', 'feature655', 'feature656',
                       'feature664', 'feature665', 'feature668', 'feature675', 'feature676', 'feature677', 'feature680',
                       'feature687', 'feature688', 'feature689', 'feature690', 'feature691', 'feature693', 'feature695',
                       'feature698', 'feature703', 'feature712', 'feature713', 'feature714', 'feature715', 'feature716',
                       'feature721', 'feature726', 'feature727', 'feature732', 'feature733', 'feature735', 'feature736',
                       'feature741', 'feature742', 'feature745', 'feature749', 'feature750', 'feature751', 'feature753',
                       'feature754', 'feature755', 'feature757', 'feature758', 'feature759', 'feature760', 'feature762',
                       'feature763', 'feature776', 'feature777', 'feature781', 'feature782', 'feature783', 'feature784',
                       'feature787', 'feature788', 'feature790', 'feature791', 'feature792', 'feature793', 'feature794',
                       'feature795', 'feature799', 'feature800', 'feature805', 'feature810', 'feature811', 'feature812',
                       'feature813', 'feature814', 'feature815', 'feature817', 'feature820', 'feature826', 'feature829',
                       'feature830', 'feature831', 'feature842', 'feature849', 'feature850', 'feature853', 'feature854',
                       'feature856', 'feature857', 'feature858', 'feature859', 'feature860', 'feature861', 'feature862',
                       'feature863', 'feature864', 'feature865', 'feature867', 'feature868', 'feature869', 'feature870',
                       'feature871', 'feature872', 'feature873', 'feature874', 'feature875', 'feature876', 'feature877',
                       'feature878', 'feature879', 'feature887', 'feature888', 'feature890', 'feature891', 'feature892',
                       'feature893', 'feature894', 'feature896', 'feature897', 'feature898', 'feature899', 'feature900',
                       'feature901', 'feature907', 'feature908', 'feature909', 'feature911', 'feature913', 'feature915',
                       'feature916', 'feature917', 'feature918', 'feature919', 'feature920', 'feature921', 'feature922',
                       'feature923', 'feature924', 'feature925', 'feature927', 'feature928', 'feature930', 'feature932',
                       'feature933', 'feature934', 'feature935', 'feature936', 'feature937', 'feature938', 'feature939',
                       'feature940', 'feature941', 'feature942', 'feature943', 'feature944', 'feature945', 'feature946',
                       'feature947', 'feature948', 'feature949', 'feature950', 'feature951', 'feature952', 'feature953',
                       'feature954', 'feature985', 'feature986', 'feature987', 'feature988', 'feature989', 'feature990',
                       'feature991', 'feature992', 'feature993', 'feature994', 'feature995', 'feature996', 'feature997',
                       'feature998', 'feature999', 'feature1000', 'feature1001', 'feature1002', 'feature1003',
                       'feature1004', 'feature1035', 'feature1036', 'feature1038', 'feature1039', 'feature1043',
                       'feature1055', 'feature1056', 'feature1057', 'feature1059', 'feature1063', 'feature1064',
                       'feature1065', 'feature1066', 'feature1067', 'feature1068', 'feature1069']

        df = None

        if (file_extension == '.csv'):
            df = pd.read_csv(file_path)
        elif (file_extension == '.parquet'):
            df = pq.read_table(file_path).to_pandas()
        else:
            print("Файл не поддерживается")
            show_warning_file()
            return

        print("Идет удаление ненужных признаков...")
        df_test = df[feature_fil]
        print("Удаление завершено\n")





        try:
            loaded_model = xgb.Booster()
            loaded_model.load_model('xgboost_after_del_f.model')
            # X_test = df_test.drop(["id", "target"], axis=1)  # Предполагается, что id является первым столбцом
            X_test = df_test.drop(["id"], axis=1)

            dtest = xgb.DMatrix(X_test)


            # Предсказание
            predictions = loaded_model.predict(dtest)
            rounded_predictions = [round(pred) for pred in predictions]

            count_of_1 = sum(rounded_predictions)
            print("Количество 1: ", count_of_1)
            print("Количество всего: ", len(predictions))

            # Создание DataFrame с предсказаниями и id
            df_predictions = pd.DataFrame(
                {'id': df_test['id'], 'target_bin': rounded_predictions, 'target_prob': predictions})
            df_predictions.to_csv(folder_path+'\\test.csv', index=False)

        except xgb.core.XGBoostError as xgb_error:
            print(f"XGBoostError: {xgb_error}")

        self.toplevel_window = ToplevelWindowBoost(self, df_predictions)



        # Закрытие предыдущего окна Toplevel, если оно открыто
        # if self.toplevel_window and self.toplevel_window.winfo_exists():
        #     self.toplevel_window.destroy()

        # Создание нового окна Toplevel






class ToplevelWindowBoost(CTkToplevel):
    def __init__(self, root, df_predictions, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set the geometry to stretch to the full height and be positioned on the right side
        self.geometry(f"600x{screen_height-100}+{screen_width - 600}+0")
        self.title("Metrics")
        self.iconbitmap("sberbank_icon-icons.com_71976.ico")

        # Create a Treeview widget with a dark color scheme
        style = ttk.Style(self)
        style.configure("Treeview", background="#2E2E2E", foreground="#FFFFFF", fieldbackground="#2E2E2E", font=('Helvetica', 10))
        style.map("Treeview", background=[("selected", "#555555")])

        self.tree = ttk.Treeview(self, columns=('id', 'target_bin', 'target_prob'), show='headings', style="Treeview")

        # Define column headings and set column width
        self.tree.heading('id', text='ID', anchor='w')
        self.tree.column('id', width=100, anchor='w')

        self.tree.heading('target_bin', text='Target Binary', anchor='w')
        self.tree.column('target_bin', width=150, anchor='w')

        self.tree.heading('target_prob', text='Target Probability', anchor='w')
        self.tree.column('target_prob', width=150, anchor='w')

        # Insert data into the treeview
        for index, row in df_predictions.iterrows():
            id_str = str(row['id'])
            self.tree.insert('', 'end', values=(id_str, row['target_bin'], row['target_prob']))

        # Add vertical scrollbar
        y_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=y_scrollbar.set)
        y_scrollbar.pack(side="right", fill="y")

        # Pack the treeview widget
        self.tree.pack(fill='both', expand=True, padx=20, pady=20)



class ToplevelWindowNeural(CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x300+9600x1200")
        self.title("Metrics")
        self.iconbitmap("sberbank_icon-icons.com_71976.ico")

        self.label = CTkLabel(self, text="Андре пошел нахуй")
        self.label.pack(padx=20, pady=20)


# Класс бутонов
# class MyRadioFrame(CTkFrame):
#     def __init__(self, master, title, values, prediction_callback):
#         super().__init__(master)
#         self.grid_columnconfigure(0, weight=1)
#         self.values = values
#         self.title = title
#         self.radio_var = tk.StringVar()
#         self.prediction_callback = prediction_callback
#
#         self.title_label = CTkLabel(self, text=self.title, fg_color="gray30", corner_radius=6)
#         self.title_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
#
#         for i, value in enumerate(self.values):
#             radio_button = CTkRadioButton(self, text=value, variable=self.radio_var, value=value, corner_radius=11)
#             radio_button.grid(row=i + 1, column=0, padx=10, pady=(10, 0), sticky="w")
#
#     def get(self):
#         return self.radio_var.get()


# инфа о коректе
def show_info(text):
    # Default messagebox for showing some information
    CTkMessagebox(title="Info", message=text)


# инфа о варе
def show_warning():
    CTkMessagebox(title="Warning Message!", message="Enter data in the fields!",
                  icon="warning", option_1="Ok")


def show_warning_file():
    CTkMessagebox(title="Warning Message!", message="file is not supported!",
                  icon="warning", option_1="Ok")


# инфа о варе с выбором ии
def show_warning_ai():
    CTkMessagebox(title="Warning Message!", message="Select AI!",
                  icon="warning", option_1="Ok")


if __name__ == "__main__":
    app = App()
    app.mainloop()