import copy

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_user_info_image(userName, anonimyzedDescription, nFollowersStr, nFollowedStr, edadEstimadaStr,
                           generoEstimadoStr, verificatStr):
    # Set up the layout parameters
    width = 700
    height = 700

    # Create a blank figure
    fig = go.Figure()

    # Add annotations for user information
    fig.add_annotation(
        text="Anàlisis de l'usuari: " + userName,
        x=0.05,
        y=4.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    # Break the description into multiple lines
    DescriptionTextManyLines = (anonimyzedDescription[:60] + '<br>' + anonimyzedDescription[60:120] + '<br>' +
                                anonimyzedDescription[120:180] + '<br>' + anonimyzedDescription[180:240] + '<br>' +
                                anonimyzedDescription[240:300] + '<br>' + anonimyzedDescription[300:])

    # description_lines = [anonimyzedDescription[i:i+60] for i in range(0, len(anonimyzedDescription), 60)]
    # description_text = '\n'.join(description_lines)
    # print(DescriptionTextManyLines)
    fig.add_annotation(
        align='left',
        text=DescriptionTextManyLines,
        arrowwidth=200,
        x=0.05,
        y=3.5,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Número de Seguidors: " + nFollowersStr,
        x=0.05,
        y=2.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Número de Seguits: " + nFollowedStr,
        x=0.05,
        y=1.75,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Edat Estimada: " + edadEstimadaStr,
        x=0.05,
        y=1.5,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Gènere Estimat: " + generoEstimadoStr,
        x=0.05,
        y=1.25,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Compte Verificat: " + verificatStr,
        x=0.05,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    # Set the layout parameters
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='white',
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False}
    )

    return fig


def draw_radarplot(categories0, valors0, etiquetes, titol, colors):
    categories1 = copy.deepcopy(categories0)
    valors1 = copy.deepcopy(valors0)  # we make a true local copy because we modify them.

    # Layout configuration
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[np.min(valors1), np.max(valors1)]  # Set the range for the radial axis
            )
        )
    )
    fig = go.Figure(layout=layout)

    #		# Create trace for the radial plot
    #	trace = go.Scatterpolar(
    #	    r=valors,
    #	    theta=categories,
    #	    fill='toself'  # Fill the area inside the plot
    #	)
    # We add the initial item at the end to close the lines.
    categories1.append(categories1[0])
    for i, xValue in enumerate(valors1):
        xValue.append(xValue[0])
        fig.add_trace(
            go.Scatterpolar(
                r=xValue,
                theta=categories1,
                mode='lines',
                line_color=colors[i],
                name=etiquetes[i],
            )
        )
    return fig


def draw_horizontal_barplot(barNames, valuesToShow0, colorsToShowValues0, etiquetes0, tittle):
    # First of all, we reverse the lists to show them in the proper order.
    valuesToShow = valuesToShow0.copy()
    colorsToShowValues = colorsToShowValues0.copy()
    etiquetes = etiquetes0.copy()

    valuesToShow.reverse()
    colorsToShowValues.reverse()
    etiquetes.reverse()

    # Set height of bar
    barHeight = 0.25
    if len(valuesToShow) == 4:
        barHeight = 0.20

    fig = go.Figure()

    # Set position of bar on Y axis
    br1 = list(range(len(valuesToShow[0])))

    # Make the plots
    for i in range(len(valuesToShow)):
        spacer = i * barHeight
        br = [x + spacer for x in br1]
        fig.add_trace(
            go.Bar(y=br, x=valuesToShow[i], orientation='h', marker_color=colorsToShowValues[i], name=etiquetes[i]))

    spacing = [r + barHeight for r in range(len(valuesToShow[0]))]
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=spacing,
            ticktext=barNames,
            automargin='width'
        ),
        legend=dict(
            traceorder="normal",
            tracegroupgap=0
        ),
        title=tittle
    )

    return fig


def draw_vertical_acumulated_barplot_plotly(barNames, valuesForEachGroup, colors, labels, title):
    fig = go.Figure()
    NumberOfBars = len(barNames)
    ind = list(range(NumberOfBars))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    pltForLegends = []
    y_offset = 0

    bottomValues = [0] * NumberOfBars
    for i in range(len(valuesForEachGroup)):
        p = fig.add_trace(go.Bar(x=ind, y=valuesForEachGroup[i], width=width, marker_color=colors[i], name=labels[i]))
        # Let's print the percentages.
        for idx, val in enumerate(valuesForEachGroup[i]):
            percent = round(val)
            if percent > 0:
                fig.add_annotation(
                    x=ind[idx], y=bottomValues[idx] + val / 2,
                    text=f'{percent}%',
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    xshift=0, yshift=y_offset
                )
        pltForLegends.append(p)
        bottomValues = [bottomValues[j] + val for j, val in enumerate(valuesForEachGroup[i])]

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis=dict(
            tickmode='array',
            tickvals=ind,
            ticktext=barNames
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=10,
            range=[0, 100]
        ),
        legend=dict(
            traceorder="normal",
            tracegroupgap=0
        )
    )
    return fig


def draw_vertical_barplot(barNames, valuesToShow, colorsToShowValues, labels, title):
    # set width of bar
    barWidth = 0.25
    if len(valuesToShow) == 4:
        barWidth = 0.20

    fig = go.Figure()

    # Set position of bar on X axis
    br1 = list(range(len(valuesToShow[0])))

    # Make the plots
    for i in range(len(valuesToShow)):
        spacer = i * barWidth
        br = [x + spacer for x in br1]
        fig.add_trace(go.Bar(x=br, y=valuesToShow[i], marker_color=colorsToShowValues[i], name=labels[i]))

    spacing = [r + barWidth for r in range(len(valuesToShow[0]))]
    fig.update_xaxes(tickvals=spacing, ticktext=barNames, tickangle=25)

    fig.update_layout(title=title, barmode='group')

    return fig


def draw_donutplot(valuesToShow, labels, tittle, tittleText):
    # print (tittle)
    width = 1.5
    # kwargs = dict(size=20, font=dict(weight='bold'), valign='middle')
    kwargs = dict(font=dict(size=20), valign='middle', align='center')
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}], [{'type': 'pie'}, {'type': 'pie'}]])

    annotations1 = [];
    # legends1 = [];
    for i in range(len(valuesToShow)):
        labelToShow = tittle[i];
        if i == 0:
            xPos = 1;
            yPos = 1;
            xCord = 0.15;
            yCord = 0.20;

        if i == 1:
            xPos = 1;
            yPos = 2;
            xCord = 0.19;
            yCord = 0.80;

        if i == 2:
            xPos = 2;
            yPos = 1;
            xCord = 0.83;
            yCord = 0.20;

        if i == 3:
            xPos = 2;
            yPos = 2;
            xCord = 0.80;
            yCord = 0.80;

        # print ('legend'+str(i+1))

        fig.add_trace(go.Pie(
            labels=labels,
            values=valuesToShow[i],
            hole=0.6,
            marker=dict(colors=["blue", 'orange']),
            textinfo='percent',
            name=tittle[i],
            domain={'x': [0.5 * (i % 2), 0.5 * (i % 2) + 0.5], 'y': [0.5 * (i // 2), 0.5 * (i // 2) + 0.5]},
            textposition='inside',
            direction='clockwise',
            showlegend=True,
            legend='legend' + str(i + 1)
        ), xPos, yPos)
        # print ("Coordinates = ",xCord, yCord, labelToShow)
        annotations1.append(dict(text=labelToShow, x=xCord, y=yCord, showarrow=False, **kwargs))
        # legends1.append(dict(xCord=0, y=-0.5, orientation='h'))

    fig.update_layout(
        annotations=annotations1,
        title={
            'text': tittleText,
            'y': 0.95,  # new
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'  # new
        },
        # legend=dict(yanchor='top', y=0.85, tracegroupgap=35))
        # legend_tracegroupgap=180
        #      legend=[dict(
        #    		yanchor="top",
        #    		y=0.99,
        #    		xanchor="left",
        #    		x=0.01
        #				),dict(
        #    		yanchor="top",
        #    		y=0.99,
        #    		xanchor="left",
        #    		x=0.55
        #				)],
        showlegend=True,
        legend1=dict(
            yanchor="bottom",
            y=0.50,
            xanchor="left",
            x=0.10
        ),
        legend2=dict(
            yanchor="bottom",
            y=0.50,
            xanchor="left",
            x=0.10
        ),

        legend3=dict(
            yanchor="top",
            y=0.70,
            xanchor="right",
            x=0.40
        ),

        legend10=dict(
            yanchor="bottom",
            y=0.55,
            xanchor="left",
            x=0.55
        )
    )

    return fig
