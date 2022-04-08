import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkhtmlview import HTMLScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import io
from wordcloud import WordCloud, STOPWORDS
from tkinter.font import BOLD, Font
import datetime as dt
import heatmapMatplotlib as hmpl

class Data():
    """Class Data kapselt die Logik für die Arbeit mit dem Originaldatensatz"""

    __error_message = ""
    __df = None # Pandas Data Frame 'Hotels Reviews' Datensatz

    def __init__(self, file_name):
        """Class Data Instanziierung
        file_name - Dateipfad
        """
        print("------Data()__init__ Started", dt.datetime.now())
        self.__file_name = file_name
        self.__read_data()
        self.__renameColumns()
        self.__check_duplicates()
        self.__create_new_Columns()

        # Erstellen der neuen Tabelle. Eine Spalte für jedes Hotel
        self.__uniq_hotels = self.__df[
            ['Hotel_Name', 'Hotel_Address', 'Additional_Number_of_Scoring', 'Average_Score', 'Total_Number_of_Reviews',
             'lat', 'lng', 'Country']].drop_duplicates()

        print("------Data()__init__ Finished", dt.datetime.now())

    def __read_data(self):
        """Datensatz (in CSV Format) Einlesen in Pandas Data Frame"""
        try:
            self.__df = pd.read_csv(self.__file_name) #, nrows = 50000)
        except Exception as ex:
            self.__error_message = f"An Exception of type {type(ex).__name__} occurred.\nArguments: {ex.args[0]}"

    def __renameColumns(self):
        """Spalte Umbenennen"""
        self.__df.rename(columns={"Review_Total_Negative_Word_Counts":"Negative_Word_Counts", "Review_Total_Positive_Word_Counts":"Positive_Word_Counts",
                                  "Total_Number_of_Reviews_Reviewer_Has_Given":"Num_of_Reviews_Reviewer_Has_Given"}, inplace=True)

    def __check_duplicates(self):
        """Prüfen, ob im Datensatz die gedoppelte Zeile geben"""
        print("duplicates = ",self.__df.duplicated().sum())
        self.__df.drop_duplicates() #löschen Duplikate

    def __create_new_Columns(self):
        """neue Spalten erstellen"""
        self.__create_Country_Column()
        self.__create_Review_Year_Month()
        self.__create_Trip_Type()
        self.__create_Days_Stayed()
        self.__create_Submitted_MD()
        self.__create_Traveler_type()
        self.__create_With_pets()
        self.__create_Review_Type()
        self.__get_top_nationalities()

        # self.__get_all_Tags()
        # self.parse_Tags()

    def __create_Country_Column(self):
        # Erstellen neue Spalte "Country"  aus Spalte "Hotel_Address"
        self.__df.Hotel_Address = self.__df.Hotel_Address.str.replace('United Kingdom','UK')
        self.__df['Country'] = self.__df.Hotel_Address.apply(lambda x: x.split(' ')[-1])
        print(f"Countries unique names: {self.__df['Country'].unique()}")

    def __create_Review_Year_Month(self):
        # Umwandeln 'Review_Date' in Datetime Objekt
        # Erstellen von neuen Spalten "Year" und "Month" aus Spalte "Review_Date"
        self.__df.Review_Date = pd.to_datetime(self.__df.Review_Date, format="%m/%d/%Y")
        self.__df['Year'] = self.__df.Review_Date.apply(lambda x: x.year)
        self.__df['Month'] = self.__df.Review_Date.apply(lambda x: x.month)

    def __create_Traveler_type(self):
        """Erstellen neue Spalte "Traveler_type" aus Spalte "Tags"""
        self.__df["Traveler_type"] = self.__df.Tags.apply(self.__get_traveler_type) #Wenden wir die Methode für jeden Wert in der Spalte "Tags" an

    def __get_traveler_type(self, tags):
        """Methode überprüft, ob die erforderliche Zeile in der 'tags' enthalten ist.
        Gibt den Wert für die Spalte "Traveler_type" zurück"""
        types = ""
        if "Couple" in tags:
            types = types + "couple"
        elif "Solo traveler" in tags:
            types = types + "solo"
        elif "Group" in tags:
            types = types + "group"
        elif "Travelers with friends" in tags:
            types = types + "with friends"
        elif "Family with young children" in tags:
            types = types + "with young children"
        elif "Family with older children" in tags:
            types = types + "with older children"
        return types

    def __create_With_pets(self):
        """Erstellen neue Spalte "With_pets" aus Spalte "Tags ('yes' oder 'no')"""
        self.__df["With_pets"] = self.__df.Tags.apply(lambda x: "yes" if "With a pet" in x else "no")

    def __create_Submitted_MD(self):
        """Erstellen neue Spalte "Submitted_Mobile_Device" aus Spalte "Tags ('yes' oder 'no')"""
        self.__df["Submitted_Mobile_Device"] = self.__df.Tags.apply(lambda x: "yes" if "Submitted from a mobile device" in x else "no")

    def __create_Trip_Type(self):
        """Erstellen neue Spalte "Trip_Type" aus Spalte "Tags ('leisure', 'business' oder None)"""
        self.__df['Trip_Type'] = self.__df.Tags.apply(self.__get_Trip_Type)

    def __get_Trip_Type(self, tags):
        """Methode überprüft, ob die erforderliche Zeile in der 'tags' enthalten ist.
                Gibt den Wert für die Spalte "Trip_Type" zurück"""
        if "Leisure trip" in tags:
            return "leisure"
        elif "Business trip" in tags:
            return "business"
        else:
            return None

    def __create_Review_Type(self):
        """Erstellen neue Spalte "Review_Type" aus Spalte "Reviewer_Score"""
        self.__df["Review_Type"]  = np.where(self.__df['Reviewer_Score'] >= 7, "positiv", "negativ")
        # self.__df["Review_Type"]  = np.where(self.__df['Positive_Word_Counts'] > self.__df['Negative_Word_Counts'], "positiv", "negativ")
        # self.__df["Review_Type"] = (self.__df['Positive_Word_Counts'] > self.__df['Negative_Word_Counts']).astype(int)

    def __create_Days_Stayed(self):
        """Erstellen neue Spalte "Days_Stayed" aus Spalte "Tags"""
        self.__df['Days_Stayed'] = self.__df.Tags.apply(self.__get_days)

    def __get_days(self, row):
        """Methode überprüft, ob die erforderliche Zeile in der 'tags' enthalten ist.
                        Gibt die Zahl von Tagen zurück"""
        n1 = row.find("Stayed")

        if n1 == -1:
            return None
        else:
            try:
                n2 = row.find("night", n1) if row.find("nights", n1) == -1 else row.find("nights", n1)
                return int(row[n1+6:n2].strip())
            except Exception as ex:
                print(row, n1, n2)
                return None

    def __get_all_Tags(self):
        # Nehmen wir einzigartige Werte aus der Spalte Tags und speichern wir sie in einer Datei
        # Datei wird für Datensatz Vorbereitung benutzt

        self.__tags_cloud = set()
        for i, row in self.__df.iterrows():
            l = row["Tags"].strip('[]').split(',')
            self.__tags_cloud.update(map(lambda x: x.strip().strip("'").strip() + '\n', l))

        try:
            with open("Tags.txt", "w") as f:
                f.writelines(self.__tags_cloud)
            print(f"new file {f.name} created ")
        except Exception as ex:
            print(f"An Exception of type {type(ex).__name__} occurred.\nArguments: {ex.args[0]}")

    def __get_top_nationalities(self):
        """Methode gibt fünf Nationalitäten zurück, die die meisten Rezensionen schrieben haben"""

        tmp = self.__df.value_counts("Reviewer_Nationality")
        top_nationalities = tmp.index[0:5]

        self.__df["Top_nationality"] = self.__df["Reviewer_Nationality"].apply(lambda x: x if x in top_nationalities else None)

    def get_statistics(self, include):
        """Gibt eine Tabelle mit Statistiken zurück
        include - Spaltentyp (int, float, object)
        """
        return self.__df.describe(include=include)

    def get_info(self):
        """Gibt Informationen zu den Datentypen in dem Dataset als String zurück"""
        buf = io.StringIO()
        self.__df.info(buf=buf)
        return buf.getvalue()

    def get_dataFrame(self, nrows=None):
        """Public Methode gibt den Datensatz zurück
        nrow - die Zahl von ersten Spalten. wenn nrows != None, dann es wird nur 'nrows' esten Spalten zurückgegeben.
        """
        return self.__df if nrows is None else self.__df.head(nrows)

    def get_uniq_Hotels(self, nrows=None):
        """Public Methode gibt den Datensatz mit einzigartige Hotels zurück
            wenn nrows != None, dann es wird nur 'nrows' esten Spalten zurückgegeben.
                """
        return self.__uniq_hotels if nrows is None else self.__uniq_hotels.head(nrows)

    def parse_Tags(self):
        """Methode wandelt die Spalte 'Tags' zu Liste um"""
        self.__df.Tags = self.__df.Tags.apply(self.get_Tags)

    def get_Tags(self, row):
        l = row.strip('[]').split(',')
        return list(map(lambda x: x.strip().strip("'").strip(), l))

    @property
    def error_message(self):
        return self.__error_message

class Plotwindow:
    """Class Data kapselt die Logik für die Erstellung der Matplotlib Plots"""
    __colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:grey', 'tab:pink', 'tab:olive'] # die Farbe, die bie Plot Darstellung benutzt werden

    def __init__(self, masterframe, ):
        """Instanziierung des Classes Plotwindow"""
        # self.figure, self.axes = plt.subplots(constrained_layout=True)
        # plt.rcParams['figure.constrained_layout.use'] = True

        self.figure = plt.Figure()
        self.figure.set_tight_layout(True)
        # self.figure.set_constrained_layout(True)

        # create canvas as matplotlib drawing area
        self.canvas = FigureCanvasTkAgg(self.figure, master=masterframe)
        self.canvas.get_tk_widget().pack(side=tk.TOP, anchor = "w", fill=tk.BOTH,
                                         expand = True, padx=5, pady=5)  # Get reference to tk_widget

    def linearPlot(self, data, columnNamex, columnNamey, columnNameGr):
        """Methode erstellt Linear Plot
        data - dateFrame mit Spalten, die geplottet werden
         columnNamex - der Name von x-Axes
         columnNamey - der Name von x-Axes
        columnNameGr - der Name der Spalte, nach der die Daten gruppiert werden
        """
        my_axes = self.figure.add_subplot(111) #Erzeugung von Axes

        if columnNameGr != '': # Mit Gruppierung
            groups = data[columnNameGr].unique() # Erhalten einzigartig Gruppenwerte

            for i, gr in enumerate(groups): #für jede Gruppe esrstellen Linear Plot (alle Linien auf gleichen Axes)
                df = data.loc[data[columnNameGr] == gr].groupby(columnNamex)[columnNamey].agg({"min", "max", "mean"})
                # Innerhalb der Gruppe erhalten wir für jeden x-Wert den Durchschnitts-, Minimal- und Maximalwert von y.
                df.reset_index(inplace=True)
                my_axes.plot(df[columnNamex], df["mean"], linewidth=1.5, color = self.__colors[i], label=gr)

            my_axes.legend()
        else: # Ohne Gruppierung
            df = data.groupby(columnNamex)[columnNamey].agg({"min", "max", "mean"})
            df.reset_index(inplace=True)
            my_axes.plot(df[columnNamex], df["mean"], linewidth=1.5)

        my_axes.set_xlabel(columnNamex)
        my_axes.set_ylabel(f"Mean Value of '{columnNamey}'")

        my_axes.minorticks_on()

        my_axes.grid(axis="x", which='major', alpha=0.3)
        my_axes.grid(axis="y", which='both', alpha=0.3)

        my_axes.set_title(f"Linear Plot", family='monospace')
        self.canvas.draw()

    def barPlotwithGroupBy(self, x, columnName):
        """Methode erstellt Bar Plot
            x - Werte, die geplottet werden
         columnName - der Name von Variable
        """
        my_axes = self.figure.add_subplot(111)

        title = f"Total number of reviews by {columnName}"
        df = pd.value_counts(x) #Geben Sie eine Serie zurück, die die Anzahl der eindeutigen Werte enthält.

        n = 25
        if len(df) > n: #Es gibt zu viel Wert für die 'Reviewer_Nationality' und "Hotel_Name" Spalte, nur die ersten 25 werden angezeigt

            rects1 = my_axes.barh(df.index[0:n], df.values[0:n])
            title = f"{title} (Top {n})"
        else:
            rects1 = my_axes.barh(df.index, df.values) #Bar plot

        my_axes.set_title(title, family='monospace')
        my_axes.set_xlabel("Total number of reviews")
        my_axes.set_ylabel(columnName)

        my_axes.grid(axis="x", which='both', alpha=0.3)

        # self.figure.tight_layout()

        self.canvas.draw()

    def histogrammPlot(self, x, columnName):
        """Methode erstellt Histogramm ohne Gruppierung
        x - Werte, die geplottet werden
        columnName - der Name von Variable"""
        my_axes = self.figure.add_subplot(111)

        my_axes.hist(x, bins = 20)#, density=True, stacked=True) #plotten Histogramm
        """density - If True, draw and return a probability density: each bin will display the bin's raw count divided by the total
         number of counts and the bin width (density = counts / (sum(counts) * np.diff(bins))),
          so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1).
        If stacked is also True, the sum of the histograms is normalized to 1."""

        my_axes.set_title("Histogram", family='monospace')

        my_axes.set_xlabel(columnName)

        my_axes.minorticks_on()

        my_axes.grid(axis="x", which='both', alpha=0.3)
        my_axes.grid(axis="y", which='both', alpha=0.3)

        self.canvas.draw()

    def histogrammGroupPlot2(self, data, columnName): #all Plots together. Not Used
        my_axes = self.figure.add_subplot(111)

        groups = data.iloc[:, 1].unique()
        n = len(groups)

        kwargs = dict(alpha=0.4, bins=25)#, density=True, stacked=True)
        for i, gr in enumerate(groups):
            my_axes.hist(data.loc[data.iloc[:, 1] == gr].iloc[:, 0], **kwargs, color = self.__colors[i], label=gr)
        my_axes.legend()

        my_axes.set_xlabel(columnName)
        my_axes.set_ylabel("Total number of reviews")

        my_axes.grid(axis="x", which='both', alpha=0.3)
        my_axes.grid(axis="y", which='both', alpha=0.3)

        my_axes.minorticks_on()

        my_axes.set_title("Histogram", family='monospace')

        self.canvas.draw()

    def histogrammGroupPlot(self, data, columnName):
        """Methode erstellt Histogramm ohne Gruppierung
        data - Werte, die geplottet werden zusammen mit Gruppen
        columnName - der Name von Variable"""
        groups = list(data.iloc[:, 1].unique())

        min_x = min(data[columnName])
        max_x = max(data[columnName])

        kwargs = dict(bins=20, density=True, stacked=True)

        if None in groups:
            groups.remove(None)
        n = len(groups)

        min_y = 0
        max_y = 0

        for i, gr in enumerate(groups):
            my_axes = self.figure.add_subplot(1, n, i+1)
            my_axes.hist(data.loc[data.iloc[:, 1] == gr].iloc[:, 0], **kwargs, color=self.__colors[i], label=gr)

            my_axes.set_xlim([min_x, max_x])
            t_min_y, t_max_y = my_axes.get_ylim()

            min_y = t_min_y if i == 0 else min(t_min_y, min_y)
            max_y = max(t_max_y, max_y)

            my_axes.legend()

            my_axes.set_xlabel(columnName)
            my_axes.set_ylabel("Total number of reviews")

            my_axes.grid(axis="x", which='both', alpha=0.3)
            my_axes.grid(axis="y", which='both', alpha=0.3)

            my_axes.minorticks_on()

        all_axes = self.figure.get_axes()
        for a in all_axes:
            a.set_ylim([min_y, max_y])
        # my_axes.set_title("Histogram", family='monospace')

        self.canvas.draw()

    def scatterPlot(self, x, y, columnNamex, columnNamey):
        """Methode erstellt Scatter Plot
        x - x-Werte, die geplottet werden
        y - y-Werte, die geplottet werden
        columnNamex - der Name von x - Variable
        columnNamey - der Name von y - Variable
        """
        my_axes = self.figure.add_subplot(111)
        my_axes.scatter(x, y, s=3, marker="o", alpha = 0.5)
        my_axes.set_xlabel(columnNamex)
        my_axes.set_ylabel(columnNamey)
        my_axes.set_title("Scatter Plot", family='monospace')

        my_axes.grid(axis="x", which='both', alpha=0.3)
        my_axes.grid(axis="y", which='both', alpha=0.3)

        my_axes.minorticks_on()

        self.canvas.draw()

    def heatmap_plot(self, df):
        # """Methode erstellt Korrelation Matrix
        # df - dataFrame mit x und y Werte
        # """
        my_axes = self.figure.add_subplot(111)

        im, cbar = hmpl.heatmap(df, df.columns.values, df.columns.values, ax=my_axes, cbarlabel="Correlation Coefficient", cmap=plt.cm.Blues)
        hmpl.annotate_heatmap(im, valfmt="{x:.2f}")
        # self.figure.tight_layout()
        # https: // matplotlib.org / stable / tutorials / intermediate / tight_layout_guide.html

        my_axes.set_title("Correlation between the variables", family='monospace')

        l, b, w, h = my_axes.get_position().bounds
        ll, bb, ww, hh = cbar.ax.get_position().bounds

        my_axes.set_position([l - 0.05, 0.01, w, h])
        cbar.ax.set_position([ll - 0.05, 0.01, ww, hh])

        self.canvas.draw()

    def wordCloud_plot(self, x, y):
        """Methode erstellt Word Cloud Plot
            x - negative Bewertungen
            y - positive Bewertungen
        """
        my_axes1 = self.figure.add_subplot(121)
        my_axes2 = self.figure.add_subplot(122)

        text1 = " ".join(review for review in x)
        text2 = " ".join(review for review in y)

        stopwords = set(STOPWORDS)
        stopwords.update(["Negative", "Positive", "Nothing", "hotel", "location", "room", "staff", "breakfast", "rooms", "London"])

        wordcloud1 = WordCloud(stopwords=stopwords, max_font_size=50, max_words=100, background_color="white").generate(text1)
        wordcloud2 = WordCloud(stopwords=stopwords, max_font_size=50, max_words=100, background_color="white").generate(text2)

        my_axes1.imshow(wordcloud1.recolor(color_func=hmpl.blue_color_func, random_state=3), interpolation="bilinear")
        my_axes2.imshow(wordcloud2.recolor(color_func=hmpl.red_color_func, random_state=3), interpolation="bilinear")

        my_axes1.set_title("Word Cloud 'Negative Reviews'", family='monospace', pad = 15)
        my_axes2.set_title("Word Cloud 'Positive Reviews'", family='monospace', pad = 15)

        my_axes1.axis("off")
        my_axes2.axis("off")

        self.canvas.draw()

    def boxPlot(self, data, columnNamex, columnNameGr):
        """Methode erstellt Box Plot
                    data - Werte, die geplottet werden, zusammen mit Gruppen
                 columnNamex - der Name von x-Variable
                 columnNameGr - der Name von Gruppe Variable
                """
        #https://stackoverflow.com/questions/16090241/pandas-dataframe-as-input-for-matplotlib-pyplot-boxplot
        my_axes = self.figure.add_subplot(111)
        if columnNameGr != '':
            groups = list(data[columnNameGr].unique())
            if None in groups:
                groups.remove(None)

            l = [data.loc[data[columnNameGr] == gr][columnNamex] for gr in groups]
            bplot = my_axes.boxplot(l,
                            vert=True,  # vertical box alignment
                            patch_artist=True, labels=groups)
        else:
            bplot = my_axes.boxplot(data,
                             vert=True,  # vertical box alignment
                             patch_artist=True, labels = [columnNamex])#,  # fill with color
                            # labels=groups)  # will be used to label x-ticks

        for patch, color in zip(bplot['boxes'], self.__colors):
            patch.set_facecolor(color)

        my_axes.set_ylabel('Value')
        my_axes.set_title(f"Box Plot '{columnNamex}'", family='monospace')
        self.canvas.draw()

    def clearplot(self):
        # entfernt alle Axes
        all_axes = self.figure.get_axes() # get alle Axes auf Figure
        for a in all_axes:
            a.remove() # clear axes
        self.canvas.draw()

class Application(tk.Tk):
    """Class Application kapselt die Logik für die Erstellung der Tkinter Widgets"""
    __groups = ['', 'Country', "Review_Type", 'Submitted_Mobile_Device', 'Top_nationality', 'Traveler_type', 'Trip_Type', 'Year'] #Group Variable
    __no_group_columns = ['Average_Score', 'Additional_Number_of_Scoring', 'Total_Number_of_Reviews', ]

    def __init__(self):
        """Instanziierung Methode für Application Class"""
        print("------Application()__init__ Start", dt.datetime.now())
        tk.Tk.__init__(self)
        self.title("Data visualisation '515K Hotel Reviews Data in Europe'")
        self.geometry("1100x750")
        self.__data = Data("Hotel_Reviews.csv")
        self.__create_widgets()
        print("------Application()__init__ Finished", dt.datetime.now())

    def __create_widgets(self):
        """Erstellt alle Applikation Widgets"""
        self.container = tk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand = True)
        nb = ttk.Notebook(self.container)
        tab1 = ttk.Frame(nb)
        tab2 = ttk.Frame(nb)
        tab3 = ttk.Frame(nb)
        tab4 = ttk.Frame(nb)
        tab5 = ttk.Frame(nb)
        tab6 = ttk.Frame(nb)
        tab7 = ttk.Frame(nb)
        tab8 = ttk.Frame(nb)
        tab9 = ttk.Frame(nb)
        tab10 = ttk.Frame(nb)
        tab11 = ttk.Frame(nb)
        tab12 = ttk.Frame(nb)
        nb.add(tab1, text="Description")
        nb.add(tab2, text="Data Set")
        nb.add(tab6, text="Unique Hotels")
        nb.add(tab8, text="Information")
        nb.add(tab7, text="Statistics")
        nb.add(tab3, text="Bar Plot")
        nb.add(tab4, text="Histogram")
        nb.add(tab11, text="Linear Plot")
        nb.add(tab5, text="Scatter Plot")
        nb.add(tab9, text="Correlation")
        nb.add(tab12, text="Box Plot")
        nb.add(tab10, text="Word Cloud")
        nb.pack(side = tk.TOP, fill=tk.Y, expand = True)
        self.__create_Description_Tab(tab1)
        self.__create_Table_Tab(tab2)
        self.__create_Info_Tab(tab8)
        self.__create_unique_hotels_Tab(tab6)
        self.__create_Bar_Plot_Tab(tab3)
        self.__create_Histogramm_Tab(tab4)
        self.__create_Scatter_Plot_Tab(tab5)
        self.__create_stat_Tab(tab7)
        self.__create_correlation_Tab(tab9)
        self.__createWordcloud_Tab(tab10)
        self.__createLinear_Plot_Tab(tab11)
        self.__createBox_Plot_Tab(tab12)

        self.statusBar = tk.Label(self.container, text="Ready")
        self.statusBar.pack(anchor = "s", side="left",fill = tk.X, expand = False)

        print("------All Widgets Created", dt.datetime.now())

    def __createBox_Plot_Tab(self, container):
        """Erstellt Box Plot Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        self.__box_plot_col = tk.StringVar()
        self.__box_plot_group = tk.StringVar()

        self.__box_plot_label_col = tk.Label(container, text="Column")
        self.__box_plot_label_col.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__box_plot_cb = ttk.Combobox(container, textvariable=self.__box_plot_col, state="readonly")
        self.__box_plot_cb.pack(side='top', anchor="w", padx=5, fill=tk.X)
        self.__box_plot_cb['values'] = self.__no_group_columns + ['Negative_Word_Counts', 'Positive_Word_Counts',
                                                                  'Num_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score']
        self.__box_plot_cb.current(0)
        self.__box_plot_cb.bind("<<ComboboxSelected>>", self.__box_plot_redraw)

        self.__box_plot_label_group = tk.Label(container, text="Group")
        self.__box_plot_label_group.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__box_plot_cb_group = ttk.Combobox(container, textvariable=self.__box_plot_group)
        self.__box_plot_cb_group.pack(side='top', anchor="w", padx=5, fill=tk.X)
        self.__box_plot_cb_group['values'] = self.__groups
        self.__box_plot_cb_group.current(0)
        self.__box_plot_cb_group.bind("<<ComboboxSelected>>", self.__box_plot_redraw)
        self.__box_plot_cb_group['state'] = "disabled"

        self.__box_plot_w = Plotwindow(container)
        self.__box_plot()

        print("------Box_Plot_Tab Created", dt.datetime.now())

    def __box_plot_redraw(self, event):
        """Callback Funktion für Comboboxes auf Box Plot Tab"""
        self.__set_group_cb_state(self.__box_plot_col.get(), self.__box_plot_cb_group)

        self.__box_plot_w.clearplot()
        self.__box_plot()

    def __box_plot(self):
        """Gesammelt Daten für Box Ploterstellung"""
        x = self.__box_plot_col.get()
        gr = self.__box_plot_group.get()
        if gr != "":
            self.__box_plot_w.boxPlot(self.__data.get_dataFrame()[[x, gr]], x, gr)
        else:
            self.__box_plot_w.boxPlot(self.__data.get_dataFrame()[x], x, gr)

    def __createLinear_Plot_Tab(self, container):
        """Erstellt Linear Plot Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        self.__linear_plot_col_x = tk.StringVar()
        self.__linear_plot_col_y = tk.StringVar()
        self.__linear_plot_group = tk.StringVar()
        self.__linear_plot_columns_y = ("Negative_Word_Counts",
                                        "Positive_Word_Counts", "Reviewer_Score")

        self.__linear_plot_label_x = tk.Label(container, text="X-Column")
        self.__linear_plot_label_x.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__linear_plot_cb_x = ttk.Combobox(container, textvariable=self.__linear_plot_col_x, state="readonly")
        self.__linear_plot_cb_x.pack(side='top', anchor="w", padx=5, fill=tk.X)
        self.__linear_plot_cb_x['values'] = ("Month", 'Days_Stayed', 'Review_Date')
        self.__linear_plot_cb_x.current(0)
        self.__linear_plot_cb_x.bind("<<ComboboxSelected>>", self.__linear_plot_redraw)

        self.__linear_plot_label_y = tk.Label(container, text="Y-Column")
        self.__linear_plot_label_y.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__linear_plot_cb_y = ttk.Combobox(container, textvariable=self.__linear_plot_col_y, state="readonly")
        self.__linear_plot_cb_y.pack(side='top', anchor="w", padx=5, fill=tk.X)
        self.__linear_plot_cb_y['values'] = self.__linear_plot_columns_y
        self.__linear_plot_cb_y.current(0)
        self.__linear_plot_cb_y.bind("<<ComboboxSelected>>", self.__linear_plot_redraw)

        self.__linear_plot_label_group = tk.Label(container, text="Group")
        self.__linear_plot_label_group.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__linear_plot_cb_group = ttk.Combobox(container, textvariable=self.__linear_plot_group, state="readonly")
        self.__linear_plot_cb_group.pack(side='top', anchor="w", padx=5, fill=tk.X)
        self.__linear_plot_cb_group['values'] = self.__groups
        self.__linear_plot_cb_group.current(0)
        self.__linear_plot_cb_group.bind("<<ComboboxSelected>>", self.__linear_plot_redraw)

        self.__linear_plot_w = Plotwindow(container)
        self.__linear_plot()

        print("------Linear_Plot_Tab Created", dt.datetime.now())

    def __linear_plot_redraw(self, event):
        self.__linear_plot_w.clearplot()
        self.__linear_plot()

    def __linear_plot(self):
        x = self.__linear_plot_col_x.get()
        y = self.__linear_plot_col_y.get()
        gr = self.__linear_plot_group.get()

        if gr == '':
            df = self.__data.get_dataFrame()
            self.__linear_plot_w.linearPlot(df[[x, y]], x, y, gr)
        else:
            self.__linear_plot_w.linearPlot(self.__data.get_dataFrame()[[x, y, gr]],x, y, gr)

    def __createWordcloud_Tab(self, container):
        """Erstellt Word Cloud Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        self.__wordCloud1_w = Plotwindow(container)
        self.__wordCloud1_w.wordCloud_plot(self.__data.get_dataFrame()["Negative_Review"], self.__data.get_dataFrame()["Positive_Review"])

        print("------Wordcloud_Tab Created", dt.datetime.now())

    def __create_correlation_Tab(self, container):
        """Erstellt Correlation Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        df_corr = self.__data.get_dataFrame().corr()
        self.__heatmap_w = Plotwindow(container)
        self.__heatmap_w.heatmap_plot(df_corr)

        print("------correlation_Tab Created", dt.datetime.now())

    def __create_Description_Tab(self, container):
        """Erstellt Description Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        self.descr_text = HTMLScrolledText(container, state="disabled")
        self.descr_text.set_html(self.__get_description())
        self.descr_text.pack(side="left", fill=tk.BOTH, expand=True, padx=5, pady=5)

        print("------Description_Tab Created", dt.datetime.now())

    def __create_Info_Tab(self, container):
        """Erstellt Info Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        self.description_scroll_bar_v = tk.Scrollbar(container, orient="vertical")

        self.info_text = tk.Text(container)
        self.info_text.configure(yscrollcommand=self.description_scroll_bar_v.set)
        self.info_text.pack(side="left", fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_text.insert(1.0, self.__data.get_info())
        self.info_text.config(state="disabled")

        self.description_scroll_bar_v.config(command=self.info_text.yview)
        self.description_scroll_bar_v.pack(side="right", fill=tk.Y)

        print("------Info_Tab Created", dt.datetime.now())

    def __create_stat_Tab(self, container):
        """Erstellt Statistiks Tab
        container - der Tab, in den alle Widgets werden gelandet"""
        self.info1_scroll_bar_h = tk.Scrollbar(container, orient="horizontal")
        self.info2_scroll_bar_h = tk.Scrollbar(container, orient="horizontal")

        self.stat_label = tk.Label(container, text="Statistics for numeric columns", font = Font(size=10, weight=BOLD))
        self.stat_label.pack(side="top", anchor="w", padx=5, pady=5, expand=False)
        # self.stat_label.config(font=("Arial", 12))

        self.create_numeric_stat_tabel(container)
        self.info1_scroll_bar_h.config(command=self.info1_tree.xview)
        self.info1_scroll_bar_h.pack(side='top', fill=tk.X)

        self.stat_label = tk.Label(container, text="Statistics for object columns", font = Font(size=10, weight=BOLD))
        self.stat_label.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.create_object_stat_tabel(container)
        self.info2_scroll_bar_h.config(command=self.info2_tree.xview)
        self.info2_scroll_bar_h.pack(side='top', fill=tk.X)

        print("------stat_Tab Created", dt.datetime.now())

    def __create_Table_Tab(self, container):
        """Erstellt Tab für Datensatz Darstellung
        container - der Tab, in den alle Widgets werden gelandet"""
        self.scroll_bar_h = tk.Scrollbar(container, orient="horizontal")
        self.scroll_bar_h.pack(side='bottom', fill=tk.X)

        self.scroll_bar_v = tk.Scrollbar(container, orient="vertical")
        self.scroll_bar_v.pack(side='right', fill=tk.Y)

        self.create_table(container)

        self.scroll_bar_v.config(command=self.tree.yview)
        self.scroll_bar_h.config(command=self.tree.xview)

        print("------Table_Tab Created", dt.datetime.now())

    def __create_unique_hotels_Tab(self, container):
        self.scroll_bar_uh_h = tk.Scrollbar(container, orient="horizontal")
        self.scroll_bar_uh_h.pack(side='bottom', fill=tk.X)

        self.scroll_bar_uh_v = tk.Scrollbar(container, orient="vertical")
        self.scroll_bar_uh_v.pack(side='right', fill=tk.Y)

        self.create_uh_tabel(container)

        self.scroll_bar_uh_v.config(command=self.tree_uh.yview)
        self.scroll_bar_uh_h.config(command=self.tree_uh.xview)

        print("------unique_hotels_Tab Created", dt.datetime.now())

    def __create_Bar_Plot_Tab(self, container):
        self.__bar_plot_columns = tk.StringVar()
        self.__l_bar_plot_columns = ('Country', 'Days_Stayed', "Hotel_Name", 'Month',
                                     'Reviewer_Nationality', "Review_Type",
                                     'Submitted_Mobile_Device', 'Traveler_type', 'Trip_Type', 'With_pets', 'Year')


        self.__bar_plot_label = tk.Label(container, text = "X-Column")
        self.__bar_plot_label.pack(side="top", anchor="w", padx=5, pady=5, expand = False)

        self.__bar_plot_cb = ttk.Combobox(container, textvariable = self.__bar_plot_columns, state="readonly")
        self.__bar_plot_cb['values'] = self.__l_bar_plot_columns
        self.__bar_plot_cb.current(0)
        self.__bar_plot_cb.bind("<<ComboboxSelected>>", self.__bar_plot_redraw)
        self.__bar_plot_cb.pack(side='top', anchor="w", fill=tk.X, padx=5)

        self.__bar_plot_w = Plotwindow(container)
        self.__bar_plot()

        print("------Bar_Plot_Tab Created", dt.datetime.now())

    def __bar_plot_redraw(self, event):
        self.__bar_plot_w.clearplot()
        self.__bar_plot()

    def __bar_plot(self):
        name = self.__bar_plot_columns.get()
        self.__bar_plot_w.barPlotwithGroupBy(self.__data.get_dataFrame()[name], name)

    def __create_Histogramm_Tab(self, container):
        self.__histogramm_columns = tk.StringVar()
        self.__histogramm_group = tk.StringVar()
        self.__l_histogramm_columns = self.__no_group_columns + ["Reviewer_Score", "Negative_Word_Counts",
                                                                 "Positive_Word_Counts",
                                                                 "Num_of_Reviews_Reviewer_Has_Given"]

        self.__histogramm_label = tk.Label(container, text="X-Column")
        self.__histogramm_label.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__histogramm_cb = ttk.Combobox(container, textvariable=self.__histogramm_columns, state="readonly")
        self.__histogramm_cb['values'] = self.__l_histogramm_columns
        self.__histogramm_cb.current(0)
        self.__histogramm_cb.bind("<<ComboboxSelected>>", self.__histogramm_redraw)
        self.__histogramm_cb.pack(side='top', anchor="w", fill=tk.X, padx=5)

        self.__histogramm_label_gr = tk.Label(container, text="Group")
        self.__histogramm_label_gr.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__histogramm_group_cb = ttk.Combobox(container, textvariable=self.__histogramm_group, state="disabled")
        self.__histogramm_group_cb['values'] = self.__groups
        self.__histogramm_group_cb.current(0)
        self.__histogramm_group_cb.bind("<<ComboboxSelected>>", self.__histogramm_redraw)
        self.__histogramm_group_cb.pack(side='top', anchor="w", fill=tk.X, padx=5)


        self.__histogramm_w = Plotwindow(container)
        self.__histogramm_plot()

        print("------Histogramm_Tab Created", dt.datetime.now())

    def __histogramm_redraw(self, event):
        self.__set_group_cb_state(self.__histogramm_columns.get(), self.__histogramm_group_cb)

        self.__histogramm_w.clearplot()
        self.__histogramm_plot()

    def __set_group_cb_state(self,colName, cb):
        if colName in self.__no_group_columns:
            cb['state'] = 'disabled'
            cb.current(0)
        else:
            cb['state'] = 'readonly'

    def __histogramm_plot(self):
        name = self.__histogramm_columns.get()
        group = self.__histogramm_group.get()
        if name in self.__no_group_columns:
             self.__histogramm_w.histogrammPlot(self.__data.get_uniq_Hotels()[name], name)
        else:
            if group == "":
                self.__histogramm_w.histogrammPlot(self.__data.get_dataFrame()[name], name)
            else:
                self.__histogramm_w.histogrammGroupPlot(self.__data.get_dataFrame()[[name, group]], name)

    def __create_Scatter_Plot_Tab(self, container):
        self.__scatter_plot_col_x = tk.StringVar()
        self.__scatter_plot_col_y = tk.StringVar()

        # self.__l_scatter_plot_columns = ("Average_Score", "Total_Number_of_Reviews", "Reviewer_Score"
        #                                  "Negative_Word_Counts",
        #                                  "Positive_Word_Counts", 'lat', 'lng')

        self.__l_scatter_plot_columns = ("Additional_Number_of_Scoring", "Average_Score", "Days_Stayed", "lat", "lng", "Month",
                                         "Reviewer_Score", "Negative_Word_Counts", "Positive_Word_Counts",
                                         "Total_Number_of_Reviews", "Num_of_Reviews_Reviewer_Has_Given", "Year")

        self.__scatter_plot_label_x = tk.Label(container, text="X-Column")
        self.__scatter_plot_label_x.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__scatter_plot_cb_x = ttk.Combobox(container, textvariable=self.__scatter_plot_col_x, state="readonly")
        self.__scatter_plot_cb_x.pack(side='top', anchor="w", padx=5,fill=tk.X)
        self.__scatter_plot_cb_x['values'] = self.__l_scatter_plot_columns
        self.__scatter_plot_cb_x.current(0)
        self.__scatter_plot_cb_x.bind("<<ComboboxSelected>>", self.__scatter_plot_redraw)

        self.__scatter_plot_label_y = tk.Label(container, text="Y-Column")
        self.__scatter_plot_label_y.pack(side="top", anchor="w", padx=5, pady=5, expand=False)

        self.__scatter_plot_cb_y = ttk.Combobox(container, textvariable=self.__scatter_plot_col_y, state="readonly")
        self.__scatter_plot_cb_y.pack(side='top', anchor="w", padx=5,fill=tk.X)
        self.__scatter_plot_cb_y['values'] = self.__l_scatter_plot_columns
        self.__scatter_plot_cb_y.current(1)
        self.__scatter_plot_cb_y.bind("<<ComboboxSelected>>", self.__scatter_plot_redraw)

        self.__scatter_plot_w = Plotwindow(container)
        self.__scatter_plot()

        print("------Scatter_Plot_Tab Created", dt.datetime.now())

    def __scatter_plot_redraw(self, event):

        self.__scatter_plot_w.clearplot()
        self.__scatter_plot()

    def __scatter_plot(self):
        name_x = self.__scatter_plot_col_x.get()
        name_y = self.__scatter_plot_col_y.get()

        self.__scatter_plot_w.scatterPlot(self.__data.get_dataFrame()[name_x],
                                          self.__data.get_dataFrame()[name_y],
                                          name_x, name_y)

    def create_table(self, container):
        df = self.__data.get_dataFrame(nrows=40)
        self.tree = self.__get_tabel(container, df)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.tree.configure(xscrollcommand=self.scroll_bar_h.set)
        self.tree.configure(yscrollcommand=self.scroll_bar_v.set)

    def create_numeric_stat_tabel(self, container):
        df = self.__data.get_statistics(['int64', 'float64'])
        df["Statistics"] = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

        df = self.__last_column_to_first(df)

        self.info1_tree = self.__get_tabel(container, df)
        self.info1_tree.pack(side=tk.TOP, fill=tk.X, expand=False, padx=5, pady=5)

        self.info1_tree.configure(xscrollcommand=self.info1_scroll_bar_h.set)

    def create_object_stat_tabel(self, container):
        df = self.__data.get_statistics(['object'])
        df["Statistics"] = ['count', 'unique', 'top', 'freq']

        df = self.__last_column_to_first(df)

        self.info2_tree = self.__get_tabel(container, df)
        self.info2_tree.pack(side=tk.TOP, fill=tk.X, expand=False, padx=5, pady=5)

        self.info2_tree.configure(xscrollcommand=self.info2_scroll_bar_h.set)

    def __last_column_to_first(self, df):
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        return df[cols]

    def create_uh_tabel(self, container):
        df = self.__data.get_uniq_Hotels(nrows=40)
        self.tree_uh = self.__get_tabel(container, df)

        self.tree_uh.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree_uh.configure(xscrollcommand=self.scroll_bar_uh_h.set)
        self.tree_uh.configure(yscrollcommand=self.scroll_bar_uh_v.set)

    def __get_tabel(self, container, df):
        tree = ttk.Treeview(container)
        tree['columns'] = list(df.columns.values)
        tree.column("#0", stretch=False, width=40)
        tree.heading("#0", text="Index")
        for c in df.columns.values:
            tree.column(c, stretch=tk.YES, minwidth=5*len(c))
            tree.heading(c, text=c)
        for i, row in enumerate(df.values):
            tree.insert('', i, text=i+1, values=list(row))
        return tree

    def __get_description(self):
        file_name = "Description.txt"
        try:
            with open(file_name, "r") as f:
                return f.read()
        except FileNotFoundError as ex:
            print(f"File {file_name} not found", ex.strerror)
        except Exception as ex:
            print(f"An Exception of type {type(ex).__name__} occurred.\nArguments: {ex.args[0]}")
        return ""


app = Application()

app.mainloop()
