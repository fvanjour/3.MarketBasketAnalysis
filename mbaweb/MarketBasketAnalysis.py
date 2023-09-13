# Import libraries
import os
import pandas as pd
import dash
from dash import dcc
from dash import html
import plotly.express as px
from wordcloud import WordCloud
from io import BytesIO
import base64
from PIL import Image
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import dash_cytoscape as cyto

# Set path for data files
path = "C:\\Users\\Min Dator\\NodBootcamp\\BC#3\\Projects\\3. MarketBasketAnalysis\\data"
os.chdir(path)

# Create dataframes from CSV files
departments = pd.read_csv('departments.csv')
products = pd.read_csv('products.csv')
orders = pd.read_csv('orders.csv', low_memory=False)
market_basket_sample = pd.read_csv('market_basket_sample.csv')

# Plots

# Frequently Ordered Products - WordCloud
# Get product frequencies
product_freq = market_basket_sample['product_name'].value_counts().to_dict()

# Generate word cloud
wc = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(product_freq)
img = Image.new('RGB', (wc.width, wc.height), (255, 255, 255))
img.paste(wc.to_image())

# Convert to Base64 for displaying in Dash
buffer = BytesIO()
img.save(buffer, format='PNG')
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')

# Products Frequently Bought Together - Network graph (Cytoscape)
# Convert order data to a list of lists
transactions = market_basket_sample.groupby('order_id')['product_name'].apply(list).tolist()

# Encode the transactions into a boolean matrix
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)

# Generate frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Prepare data for the graph
nodes = []
edges = []

# Create nodes for antecedents and consequents, and edges between them
for _, row in rules.iterrows():
    antecedent = list(row['antecedents'])[0]
    consequent = list(row['consequents'])[0]

    nodes.append({"data": {"id": antecedent, "label": antecedent}})
    nodes.append({"data": {"id": consequent, "label": consequent}})
    edges.append({"data": {"source": antecedent, "target": consequent}})

elements = nodes + edges

# Product Hierarchies Based on Order Frequency - Sunburst
# Get order frequencies
order_frequencies = (market_basket_sample.groupby(['department', 'aisle', 'product_name'])
                     .size()
                     .reset_index(name='order_frequency'))

# Create the sunburst chart
fig = px.sunburst(order_frequencies,
                  path=['department', 'aisle', 'product_name'],
                  values='order_frequency')

# Orders over Time - Heatmap
# Generate a pivot table with the day of the week and hour of the day
pivot = orders.pivot_table(index='order_dow', columns='order_hour_of_day',
                           values='order_id', aggfunc='count').reset_index()

# Create heatmap
fig2 = px.imshow(pivot.drop('order_dow', axis=1),
                labels=dict(color="Order Count"),
                x=pivot.columns[1:],
                y=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])

# Popularity of Top Products over the Week
# Getting the top N products
N = 20
top_products = market_basket_sample['product_name'].value_counts().head(N).index.tolist()
filtered_data = market_basket_sample[market_basket_sample['product_name'].isin(top_products)]
day_name_mapping = {
    0: 'Sunday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
}
filtered_data['order_dow'] = filtered_data['order_dow'].map(day_name_mapping)


# Animated scatter plot
fig3 = px.scatter(filtered_data,
                 x='order_hour_of_day',
                 y='product_name',
                 animation_frame='order_dow',
                 size='order_id',
                 color='product_name',
                 category_orders={"order_dow": ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']},
                 size_max=40)

fig3.update_layout(updatemenus=[{
    'buttons': [{
        'args': [None, {'frame': {'duration': 2000, 'redraw': True}, 'fromcurrent': True}],
        'label': 'Play',
        'method': 'animate'
    },
    {
        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
        'label': 'Pause',
        'method': 'animate'
    }],
    'direction': 'left',
    'pad': {'r': 10, 't': 87},
    'showactive': False,
    'type': 'buttons',
    'x': 0.1,
    'xanchor': 'right',
    'y': 0,
    'yanchor': 'top'
}])

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([

    # Title Slide
    html.Div(
        style={
            'background': '#FFF5E1',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginBottom': '30px'
        },
        children=[
            html.H1("Market Basket Analysis", style={'textAlign': 'center'}),
            html.H3("Actionable Insights:  Unveiling Shopping Patterns", style={'textAlign': 'center'}),
            html.P("by: Franklin Vanjour", style={'textAlign': 'center', 'fontSize': '16px'})
        ]
    ),

    # Intro Slide
    html.Div(
        style={
            'background': '#FFF5E1',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginBottom': '30px'
        },
        children=[
            html.H2("Sam's Online Grocery Store", style={'textAlign': 'center'}),
            html.Ul(children=[
                html.Li("Sam has an Online Grocery Store"),
                html.Li("He wants to improve his business by deriving actionable insights from his operational data"),
                html.Li("Sam requests Frank, a Data Analyst, to help with the Analysis")
            ]),
            html.Img(src='/assets/intro.png', width="600", height="400")
        ]
    ),

    # Main Content Div
    html.Div(
        style={
            'display': 'grid',
            'gridTemplateColumns': '1fr',
            'gap': '30px',
            'background': '#E5E8E8',
            'borderRadius': '5px',
            'padding': '30px',
            'boxShadow': '0px 0px 15px rgba(0, 0, 0, 0.2)'
        },

        children=[

            # Each of the Graph/Image Divs with associated Title, Insight, and Action
            *[
                html.Div(
                    style={
                        'background': '#FFF5E1',
                        'borderRadius': '5px',
                        'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
                        'padding': '20px'
                    },
                    children=[
                        html.H2(title, style={'textAlign': 'center'}),
                        content,
                        html.H5("Insight:", style={'marginTop': '20px'}),
                        html.P(insight),
                        html.H5("Action:", style={'marginTop': '10px'}),
                        html.P(action)
                    ]
                )
                for title, content, insight, action in [

                    # Slide 1: Word Cloud Image
                    (
                        "Popular Products",
                        html.Img(
                            src=f'data:image/png;base64,{image_base64}',
                            style={'width': '400px', 'height': '400px', 'display': 'block', 'margin': 'auto'}
                        ),
                        "Find out the most ordered products.",
                        "Stock up on popular items, promote them more, or bundle them with other products."
                    ),

                    # Slide 2: Cytoscape Graph
                    (
                        "Products Bought Together",
                        cyto.Cytoscape(
                            id='cytoscape',
                            layout={'name': 'cose'},
                            elements=elements,
                            stylesheet=[
                                            {
                                                'selector': 'node',
                                                'style': {
                                                    'label': 'data(label)',
                                                    'font-size': '8px',  # Adjust based on your needs
                                                    'text-valign': 'center',
                                                    'text-halign': 'right',
                                                    'background-color': '#0074d9'
                                                }
                                            },
                                            {
                                                'selector': ':selected',
                                                'style': {
                                                    'background-color': '#FF00FF',
                                                    'font-size': '14px',
                                                    'line-color': '#FF00FF',
                                                    'target-arrow-color': '#FF00FF',
                                                    'source-arrow-color': '#FF00FF',
                                                    'text-outline-color': '#FF00FF',
                                                    'z-index':'10'
                                                }
                                            }
                                        ]

                        ),
                        "Discover which products are often bought together.",
                        "Use for cross-selling, bundling, or placing related products closer together."
                    ),

                    # Slide 3: Sunburst Graph
                    (
                        "Product Hierarchies by Order Frequency",
                        dcc.Graph(figure=fig),
                        "Identify product categories and subcategories which are most popular among customers.",
                        "Prioritize inventory and marketing efforts towards high-frequency categories and explore potential growth opportunities within less popular subcategories."
                    ),

                    # Slide 4: Heatmap
                    (
                        "Orders over Time",
                        dcc.Graph(figure=fig2),
                        "Understand the peak ordering times.",
                        "Staff up during peak times, offer promotions during off-peak hours to balance demand."
                    ),

                    # Slide 5: Animated Scatter
                    (
                        "Popularity of Top Products over Time",
                        dcc.Graph(figure=fig3),
                        "Understand the dynamic shifts in the popularity of top products over time.",
                        "Adapt inventory and marketing strategies based on dynamic product trends."
                    )
                ]
            ]

        ]  # End of Main Content Div children
    ),  # End of Main Content Div

    # Highlights/Challenges Slide
    html.Div(
        style={
            'background': '#FFFFFF',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginTop': '30px'
        },
        children=[
            html.H2("Highlights / Challenges", style={'textAlign': 'center'}),
            html.Ul(children=[
                html.Li("Learning new libraries Plotly and Dash"),
                html.Li("Identifying which:"),
                html.Ul(children=[
                    html.Li("Analysis to perform"),
                    html.Li("Plot to visualize")
                ]),
                html.Li("Resolving memory issues due to:"),
                html.Ul(children=[
                    html.Li("Multiple Interactive Plots"),
                    html.Li("Extremely Large Dataset (32 million rows)")
                ])
            ])
        ]
    )
],
style={
'background': '#F2F3F4',
'padding': '30px',
'font-family': 'Arial'
})  # End of main layout

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
