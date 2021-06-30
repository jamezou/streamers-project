
# import libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette("deep")

# load dataset
df = pd.read_csv("twitch-streamers.csv")


def home():
    st.title("Top Twitch Streamers")
    st.markdown("From June 10th, 2020 through June 10th, 2021, the top 1000 most-watched "
                "Twitch streamers were recorded in a data set that contains various measures "
                "related to their streams such as the total time the channel spent streaming "
                "and the number of followers the streamer gained for the past year. This "
                "application was made so that users can learn more about the dataset by "     
                "exploring the different variables.")

    st.markdown("Below are the explanations for each of the columns found in the dataset.")
    df_columns = pd.Series(df.columns.transpose(), name="Column")
    explanations = pd.Series(["Streamer/channel name",
                              "Total amount of time Twitch users have spent watching the streamer (in minutes)",
                              "Total amount of time the channel has spent streamed for",
                              "Highest number of viewers streamer has received",
                              "Number of viewers the streamer receives on average",
                              "Total number of followers the streamer has",
                              "Total number of follows the streamer has gained",
                              "Total number of views the streamer has gained",
                              "Whether the channel is partnered with Twitch",
                              "Whether the channel is marked for mature content",
                              "Language that the streamer streams in"], name="Explanation")

    df_explanations = pd.concat([df_columns, explanations], axis=1)
    st.table(df_explanations)


def channel_link(channel):
    base_url = 'https://www.twitch.tv/'
    start = channel.find("(")
    end = channel.find(")")
    if start != -1 and end != -1:
        channel = channel[start+1:end].lower()
    link = base_url + channel
    return link


def histogram(df_col):
    # obtain column for histogram
    column = st.sidebar.selectbox("Select a column", ['Watch time', 'Stream time', 'Peak viewers',
                                                      'Average viewers', 'Followers',
                                                      'Followers gained', 'Views gained'])
    df_col.hist(column=column)
    plt.xlabel(column)
    plt.ylabel("Count")
    st.pyplot(plt)


def comparison_with_avg(df_streamer, name):
    df_describe = df.drop(['Channel', 'Partnered', 'Mature', 'Language'], axis=1).describe().transpose()
    df_comparison = df_describe['mean']
    df_streamer = df_streamer.transpose().drop(['Channel', 'Partnered', 'Mature', 'Language'])
    df_comparison = pd.concat([df_comparison, df_streamer], axis=1)
    df_comparison.columns.values[1] = "streamer"

    # plot grouped bar chart between streamer and average stats
    df_comparison.plot(kind='bar', logy=True)
    plt.title(f"Comparison between Average and {name} Stats")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Time (in minutes) / Number of People")
    st.pyplot(plt)
    # determine if data is displayed
    view_data = st.checkbox("View comparison data")
    if view_data:
        st.dataframe(df_comparison)
    st.write("--")  # spacer


def high_lows(high_or_low, column):
    if high_or_low == "Highest":
        highest = df[column].max()
        channel = df.loc[df[column] == highest, 'Channel'].values[0]
        if column in ['Watch time', 'Stream time']:
            unit = st.sidebar.radio("Unit of Time", ['Minutes', 'Hours', 'Days'])
            if unit == "Minutes":
                st.markdown(f"Highest {column.lower()}: {highest:,} minutes")
            elif unit == "Hours":
                st.markdown(f"Highest {column.lower()}: {highest/60:,.2f} hours")
            else:
                st.markdown(f"Highest {column.lower()}: {highest/1440:,.2f} days")
            st.markdown(f"Channel: {channel}")
        else:
            st.markdown(f"Highest number of {column.lower()}: {highest:,}")
            st.markdown(f"Channel: {channel}")
    elif high_or_low == "Lowest":
        lowest = df[column].min()
        channel = df.loc[df[column] == lowest, 'Channel'].values[0]
        if column in ['Watch time', 'Stream time']:
            unit = st.sidebar.radio("Unit of Time", ['Minutes', 'Hours', 'Days'])
            if unit == "Minutes":
                st.markdown(f"Lowest {column.lower()}: {lowest:,} minutes")
            elif unit == "Hours":
                st.markdown(f"Lowest {column.lower()}: {lowest/60:,.2f} hours")
            else:
                st.markdown(f"Lowest {column.lower()}: {lowest/1440:,.2f} days")
            st.markdown(f"Channel: {channel}")
        else:
            st.markdown(f"Lowest number of {column.lower()}: {lowest:,}")
            st.markdown(f"Channel: {channel}")
    view_row = st.checkbox(f"View data for {channel}")
    if view_row:
        index = df.index[df['Channel'] == channel].tolist()
        df_channel = df.loc[index]
        st.dataframe(df_channel)
    st.markdown(f"[Click here to be redirected to Twitch channel]({channel_link(channel)})")


def overall_stats():
    df_stats = df.drop(['Channel', 'Partnered', 'Mature', 'Language'], axis=1).describe().transpose()
    df_stats = df_stats[['min', 'mean', 'max']]
    # create grouped bar chart
    df_stats.plot(kind='bar', logy=True)
    plt.title("Overall Channel Stats", fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Time (in minutes) / Number of People")
    st.pyplot(plt)


def variable_scatterplot(column1, column2):
    plt.scatter(df[column1], df[column2])
    plt.title(f"{column1} vs {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    st.pyplot(plt)


def variable_lineplot(column1, column2):
    sns.lineplot(df[column1], df[column2])
    plt.title(f"{column1} vs {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    st.pyplot(plt)


def categorical_plots(column):
    if column in ['Partnered', 'Mature']:
        sns.countplot(df[column])
        if column == "Partnered":
            plt.title("Twitch Partnered Channels", fontsize=15)
        else:
            plt.title("Channels Marked for Mature Content", fontsize=15)
        plt.xticks([0,1], labels=['False', 'True'])
    else:
        language_plot = st.sidebar.selectbox("Type of Plot", ['Bar Chart', 'Pie Chart'])
        language_count = df[column].value_counts().rename_axis(column).reset_index(name='Count')
        language_freq = language_count.groupby(column)['Count'].sum().sort_values(ascending=False)
        if language_plot == "Bar Chart":
            plt.bar(language_freq.index, language_freq)
            plt.title("Channels by Language")
            plt.xticks(rotation=45, ha='right')
        else:
            top_languages = list(language_count[column].head(9))
            for lang in language_count[column]:
                if lang not in top_languages:
                    language_count[column] = language_count[column].replace(lang, "Other")
            language_freq = language_count.groupby(column)['Count'].sum().sort_values(ascending=False)
            plt.figure(figsize=(10,6))
            plt.pie(language_freq, autopct='%1.1f%%', pctdistance=0.85)
            plt.title("Top 10 Languages Channels Stream In")
            plt.legend(labels=language_freq.index)
    st.pyplot(plt)


def narrow_data():
    # obtain user data selection
    rows = st.sidebar.radio("Select an option", ['All', 'Random Sample', 'Specific Channel', 'Highs & Lows'])
    if rows == "All":
        st.subheader("All Data")
        size = st.sidebar.select_slider("Number of rows to display", options=[100, 200, 300, 400, 500,
                                                                              600, 700, 800, 900, 1000])
        # call histogram function
        histogram(df)
        st.dataframe(df.head(size))
    elif rows == "Random Sample":
        st.subheader("Random Sample")
        # obtain user input for sample size
        sample_size = st.sidebar.slider("Sample size", 1, 1000)
        df_sample = df.sample(sample_size)
        # call histogram function
        histogram(df_sample)
        st.dataframe(df_sample)
    elif rows == "Specific Channel":
        st.subheader("Specific Channel")
        # obtain specific channel input
        channel_name = st.sidebar.text_input("Enter the channel name", "Sykkuno")
        # confirm channel exists in dataset
        if channel_name in list(df['Channel'].values):
            st.markdown(f"Viewing data for {channel_name}")
            index = df.index[df['Channel'] == channel_name].tolist()
            channel = df.loc[index]
            # call comparison to average values (grouped bar chart)
            comparison_with_avg(channel, channel_name)
            # display streamer data
            st.dataframe(channel)
        else:
            st.markdown(f"Sorry, the streamer, {channel_name} could not be found.")
            st.markdown("Please make sure you've spelled the name correctly.")
    else:
        st.subheader("Highs & Lows")
        # obtain user inputs
        choice_HL = st.sidebar.radio("Select an option", ['Lowest', 'Highest'])
        column_HL = st.sidebar.selectbox("Select a column", ['Watch time', 'Stream time', 'Peak viewers',
                                                             'Average viewers', 'Followers',
                                                             'Followers gained', 'Views gained'])
        # call function to display high/low results
        high_lows(choice_HL, column_HL)


def main():
    st.sidebar.header("Top Twitch Streamers")
    page_selection = st.sidebar.selectbox("Select a page", ['Home', 'Plotting Variables', 'EDA'])
    if page_selection == "Home":
        # call home function
        home()
    elif page_selection == "Plotting Variables":
        st.header("Plotting Variables")
        section = st.sidebar.selectbox("Select a section", ['Variable Comparison', 'Overall Comparison',
                                                            'Categorical Variables'])
        if section == "Variable Comparison":
            st.subheader("Comparison Between Variables")
            # obtain 2 column inputs
            column1_vc = st.sidebar.selectbox("First column", ['Watch time', 'Stream time', 'Peak viewers',
                                                               'Average viewers', 'Followers',
                                                               'Followers gained', 'Views gained'])
            column2_vc = st.sidebar.selectbox("Second column", ['Watch time', 'Stream time', 'Peak viewers',
                                                                'Average viewers', 'Followers',
                                                                'Followers gained', 'Views gained'])
            plot_option = st.sidebar.selectbox("Type of Plot", ['Scatterplot', 'Line Plot'])
            if plot_option == "Scatterplot":
                # call variable scatterplot function
                variable_scatterplot(column1_vc, column2_vc)
            else:
                variable_lineplot(column1_vc, column2_vc)
        elif section == "Overall Comparison":
            st.subheader("Overall Comparison")
            # call function for overall stats
            overall_stats()

        else:
            # obtain column input
            categorical_variable = st.sidebar.selectbox("Select a column", ['Partnered', 'Mature', 'Language'])
            # call function to plot categorical variable
            categorical_plots(categorical_variable)
    elif page_selection == "EDA":
        st.sidebar.subheader("Data Selection")
        st.header("Exploratory Data Analysis")
        # call view data function
        narrow_data()

    # ending remarks
    st.sidebar.markdown("--")
    st.sidebar.markdown("Created by: Jame Zou")
    st.sidebar.markdown("GitHub Repository: [Click here](https://github.com/jamezou/streamers-project)")
    st.sidebar.markdown(f"Data obtained from [SullyGnome.com](https://SullyGnome.com)")


if __name__ == "__main__":
    main()

