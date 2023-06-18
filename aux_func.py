import pandas as pd
from scipy import stats
from bokeh.io import show, output_file
from bokeh.plotting import figure, save, output_file, show
from bokeh.models import ColumnDataSource, ranges, LabelSet, Label, Range1d, PolyAnnotation, Band, Rect
from bokeh.layouts import gridplot, row, column, layout
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly
from datetime import timedelta
from bokeh.models import Range1d,ImageURL
import math
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
from bokeh.models import Span
from bokeh.models import BoxAnnotation
from bokeh.models import Title
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
from bokeh.models import Span
from bokeh.models import BoxAnnotation
from collections import OrderedDict
from io import StringIO
from math import log, sqrt
from bokeh.io import export_png
import numpy as np
import pandas as pd
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import (BasicTicker, Circle, ColumnDataSource, DataRange1d,
                          Grid, LinearAxis, PanTool, Plot, WheelZoomTool,)
from bokeh.resources import INLINE
from bokeh.sampledata.iris import flowers
from bokeh.util.browser import view
import numpy as np
from scipy import optimize, interpolate
from scipy import stats as st
from bokeh.models import Legend
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot, Text
from bokeh.models import FixedTicker,Wedge
import math
from datetime import date
from bokeh.document import Document
from bokeh.models import (Circle, ColumnDataSource, Div, Grid,
                          Line, LinearAxis, Plot, Range1d,)
from bokeh.resources import INLINE
from bokeh.util.browser import view
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings("ignore")
from bokeh.models import Arrow, NormalHead, OpenHead, VeeHead
from bokeh.plotting import figure, output_file, show
from PIL import Image
import requests
from io import BytesIO
import io
from pathlib import Path
from PIL import Image
import urllib.request
import io
import os
import random
from difflib import SequenceMatcher
from pathlib import Path
import streamlit as stream
#from streamlit_lottie import st_lottie
#from streamlit_lottie import st_lottie_spinner
import pandas as pd
import urllib, json
import requests
import aux_func
from tqdm import tqdm
import numpy as np
from math import pi
from bokeh.models.glyphs import Circle, Patches, Wedge
from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.transform import factor_cmap
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
import io
from PIL import Image
import os
import colorsys


import io
from PIL import Image
import os
import requests
#import networkx as nx

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

from colour import Color

from bokeh.models import (Arrow, ColumnDataSource, CustomJS, Label,
                        NormalHead, SingleIntervalTicker, TapTool)

def get_json_data_heatmapa(idAway_, idHome_, fix_id_):
    

    url = 'https://footstatsapiapp.azurewebsites.net//partidas/heatmapByTeam'

    data = {'idAway': idAway_, 'idChampionship': "803", 'idHome': idHome_, 'idMatch': fix_id_}

    r = requests.post(url, json=data)
    data = r.json()

    vec = []

    data_geo_position = data['data']

    home_data = data_geo_position['mandante']
    for player in home_data:
        player_id = player['idJogador']
        player_name = player['player']
        for inter_ in player['interacao']:
            inter_x = inter_['x']
            inter_y = inter_['y']
            inter_tempo = inter_['idPeriodoJogo']
            vec.append([idHome_, player_id, player_name, inter_x, inter_y, inter_tempo])


    away_data = data_geo_position['visitante']
    for player in away_data:
        player_id = player['idJogador']
        player_name = player['player']
        for inter_ in player['interacao']:
            inter_x = inter_['x']
            inter_y = inter_['y']
            inter_tempo = inter_['idPeriodoJogo']
            vec.append([idAway_, player_id, player_name, inter_x, inter_y, inter_tempo])

    df_geo_position = pd.DataFrame(data = vec, columns = ['team_id', 'player_id', 'player_name', 'inter_x', 'inter_y', 'inter_tempo'])
    df_geo_position['fix_id'] = fix_id_
    df_geo_position['inter_x'] = df_geo_position['inter_x']*105 - 52.5
    df_geo_position['inter_y'] = df_geo_position['inter_y']*(-68) + 34

    return df_geo_position

def get_color(value, max_):    
    rgb = colorsys.hsv_to_rgb(value / max_, 1.0, 1.0)
    return [round(255*x) for x in rgb]


def flatten_json(nested_json, exclude=['']):
    """Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
            exclude: Keys to exclude from output.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def get_opposing_team(rodada_num, clube_id, base_global_partidas):
    df_aux = base_global_partidas[base_global_partidas['rodada_id']==rodada_num]
    if(clube_id in df_aux['clube_casa_id'].tolist()):
        return df_aux[df_aux['clube_casa_id']==clube_id]['clube_visitante_id'].tolist()[0]
    else:
        return df_aux[df_aux['clube_visitante_id']==clube_id]['clube_casa_id'].tolist()[0]

def get_local_played_at_round(rodada_num, clube_id, base_global_partidas):
    df_aux = base_global_partidas[base_global_partidas['rodada_id']==rodada_num]
    if(clube_id in df_aux['clube_casa_id'].tolist()):
        return 'CASA'
    else:
        return 'FORA'



#%%
def get_dataframes_for_plots():

    #Coleta do DataFrame que revela o status do mercado e a rodada atual
    url_status_mercado = 'https://api.cartola.globo.com/mercado/status'
    r = requests.get(url_status_mercado)
    data = r.json()
    rodada_atual_num = data['rodada_atual']

    base_global = pd.DataFrame(columns=['atleta_id', 'scout', 'apelido', 'foto', 'pontuacao', 'posicao_id',
        'clube_id', 'entrou_em_campo', 'DS', 'FC', 'FS', 'I', 'PI', 'A', 'FD',
        'FF', 'CA', '', 'G', 'FT', 'SG', 'DE', 'GS', 'PS', 'PC', 'GC', 'CV',
        'rodada_id', 'PP', 'DP'])

    for rodada_id in tqdm(range(1,rodada_atual_num)):
        #Coleta do DataFrame de Pontuação por Rodada
        url_base_cartola = 'https://api.cartola.globo.com/atletas/pontuados/'
        dados_rodada = url_base_cartola+ '/'+ str(rodada_id)
        r = requests.get(dados_rodada)
        data = r.json()
        rodada_dataframe = pd.DataFrame.from_dict(data['atletas']).T
        rodada_dataframe = rodada_dataframe.reset_index().rename(columns={'index':'atleta_id'})
        rodada_dataframe["scout"] = rodada_dataframe["scout"].fillna({})
        rodada_dataframe_ = pd.concat([rodada_dataframe,
                                        pd.DataFrame([flatten_json(x) for x in rodada_dataframe['scout']])],
                                        axis=1)
        rodada_dataframe_ = rodada_dataframe_.fillna(0)
        rodada_dataframe_['rodada_id'] = rodada_id

        base_global = base_global.append(rodada_dataframe_)


    base_global_partidas = pd.DataFrame(columns=['partida_id', 'clube_casa_id', 'clube_casa_posicao',
        'clube_visitante_id', 'aproveitamento_mandante',
        'aproveitamento_visitante', 'clube_visitante_posicao', 'partida_data',
        'timestamp', 'local', 'valida', 'placar_oficial_mandante',
        'placar_oficial_visitante'])

    for rodada_id in tqdm(range(1,rodada_atual_num+1)):
        url_partidas = 'https://api.cartola.globo.com/partidas/'
        dados_partidas = url_partidas+ '/'+ str(rodada_id)
        r = requests.get(dados_partidas)
        data = r.json()

        df_partidas = pd.DataFrame.from_dict(data['partidas']).drop(['status_transmissao_tr',
        'inicio_cronometro_tr', 'status_cronometro_tr', 'periodo_tr',
        'transmissao'], axis=1)
        df_partidas['rodada_id'] = rodada_id

        base_global_partidas = base_global_partidas.append(df_partidas)


    base_global['opposing_team_id'] = [get_opposing_team(x, y, base_global_partidas) for x,y in zip(base_global['rodada_id'],
                                                                            base_global['clube_id'])]

    base_global['local'] = [get_local_played_at_round(x, y, base_global_partidas) for x,y in zip(base_global['rodada_id'],
                                                                            base_global['clube_id'])]


    url_mercado_jogadores = 'https://api.cartola.globo.com/atletas/mercado'
    r = requests.get(url_mercado_jogadores)
    data = r.json()

    df_clubes = pd.DataFrame.from_dict(data['clubes']).T
    df_clubes['escudo_url'] = [x['60x60'] for x in df_clubes['escudos']]


    df_posicoes = pd.DataFrame.from_dict(data['posicoes']).T
    df_status = pd.DataFrame.from_dict(data['status']).T
    df_atletas = pd.DataFrame.from_dict(data['atletas'])
    df_data_scouts = base_global

    df_data_scouts_ = pd.merge(df_data_scouts, df_clubes[['id','nome']].rename(columns={'nome':'opposing_team_name'}), how='left', left_on='opposing_team_id', right_on='id')
    df_data_scouts__ = pd.merge(df_data_scouts_, df_clubes[['id','nome']].rename(columns={'nome':'clube_name'}), how='left', left_on='clube_id', right_on='id')
    df_data_scouts___ = pd.merge(df_data_scouts__, df_posicoes[['id','abreviacao']].rename(columns={'abreviacao':'posicao_nome'}), how='left', left_on='posicao_id', right_on='id')
    df_data_scouts___ = df_data_scouts___.drop(['id_x','id_y','id'], axis=1)
    df_data_scouts___ = df_data_scouts___.fillna(0)

    df_data_scouts___ = df_data_scouts___[['rodada_id','local','atleta_id', 'apelido', 'foto', 'pontuacao', 'posicao_id', 'posicao_nome',
                                            'clube_id','clube_name', 'entrou_em_campo',  'opposing_team_id', 'opposing_team_name',
                                            'DS', 'FC', 'FS', 'I', 'PI', 'A', 'FD',
                                            'FF', 'CA', '', 'G', 'FT', 'SG', 'DE', 'GS', 'PS', 'PC', 'GC', 'CV',
                                            'PP', 'DP', 'scout']]

    df_data_scouts___['foto'] = [str(x).replace("FORMATO", '220x220') if not(pd.isnull(x)) else None for x in df_data_scouts___['foto'].tolist()]
    df_atletas['foto'] = [str(x).replace("FORMATO", '220x220') if not(pd.isnull(x)) else None for x in df_atletas['foto'].tolist()]
    df_data_scouts___['F'] = df_data_scouts___['FF'] + df_data_scouts___['FD'] + df_data_scouts___['FT']
    df_data_scouts___['class_pontuacao'] = pd.cut(df_data_scouts___['pontuacao'], [-10,0,2,5,10,50], right=True, labels=["Péssimo","Ruim","Médio","Bom","Mitou"])

    df_confrontos_rodada_atual = base_global_partidas[base_global_partidas['rodada_id']==rodada_atual_num]

    return df_confrontos_rodada_atual, df_atletas, df_data_scouts___, df_status, df_posicoes, base_global_partidas, rodada_atual_num, df_clubes

#%%
def get_dataframe_mapa_confronto(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, pos, rodada_atual_num):

    team_id = df_confrontos_rodada_atual.clube_casa_id.tolist() + df_confrontos_rodada_atual.clube_visitante_id.tolist()
    local_de_jogo_da_rodada = len(df_confrontos_rodada_atual)*['CASA'] + len(df_confrontos_rodada_atual)*['FORA']

    dataframe_local_rodada = pd.DataFrame(columns = ['clube_id','local'])
    dataframe_local_rodada['clube_id'] = team_id
    dataframe_local_rodada['local'] = local_de_jogo_da_rodada
    dataframe_local_rodada['flag_mapa'] = 1

    df_data_scouts___adjs = pd.merge(df_data_scouts___, dataframe_local_rodada, how='left', on=['clube_id','local'])
    #df_data_scouts___adjs = df_data_scouts___adjs[~pd.isnull(df_data_scouts___adjs['flag_mapa'])]
    df_data_scouts___adjs_ = df_data_scouts___adjs[df_data_scouts___adjs['rodada_id']>=(rodada_atual_num-10)]

    base_pontuacao_cedida = df_data_scouts___adjs_[df_data_scouts___adjs_['posicao_nome']==pos].groupby(['opposing_team_name','opposing_team_id'])['pontuacao'].mean().reset_index().sort_values(by='pontuacao', ascending=False)
    base_pontuacao_executada = df_data_scouts___adjs_[df_data_scouts___adjs_['posicao_nome']==pos].groupby(['clube_name','clube_id'])['pontuacao'].mean().reset_index().sort_values(by='pontuacao', ascending=False)

    df_confro_a = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]
    df_confro_b = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]

    df_confro_b['clube_casa_id'] = df_confro_a['clube_visitante_id']
    df_confro_b['clube_visitante_id'] = df_confro_a['clube_casa_id']

    df_confronto_list = df_confro_b.append(df_confro_a)
    df_confronto_list.columns = ['clube_id','opposing_team_id']

    dataframe_mapa_confronto = pd.merge(pd.merge(base_pontuacao_executada, df_confronto_list, how='left', on='clube_id'),
    base_pontuacao_cedida,
    how='left',
    on='opposing_team_id').rename(columns={'pontuacao_x':'pontuacao_executada', 'pontuacao_y':'pontuacao_cedida'})

    dataframe_mapa_confronto = pd.merge(dataframe_mapa_confronto, df_clubes[['id','escudo_url']].rename(columns={'id':'clube_id'}), how='left', on='clube_id')

    return dataframe_mapa_confronto

# %%

def get_mapa_confronto_plot(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, pos, rodada_atual_num):


    dataframe_mapa_confronto = get_dataframe_mapa_confronto(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, pos, rodada_atual_num)
    p = figure(width=600, height=600, output_backend="webgl")
    source = ColumnDataSource(dataframe_mapa_confronto)

    dict_title = dict({
        'ata': 'Atacantes',
        'lat': 'Laterais',
        'mei': 'Meio-Campo',
        'zag': 'Zagueiro',
        'gol': 'Goleiro',
        'tec': 'Técnico'
    })

    title_pos = dict_title[pos]

    p.add_layout(Title(text='MAPA DO CONFRONTO | {0} | RODADA {1}'.format(title_pos.upper(), rodada_atual_num),
     align="center"), "above")

    image2 = ImageURL(url="escudo_url",
                    x='pontuacao_cedida',
                    y='pontuacao_executada',
                    w_units='screen',
                    h_units='screen',
                    w = 60,
                    h = 60,
                    anchor="center")

    p.add_glyph(source, image2)

    p.yaxis.axis_label = 'Pontuação Conquistada no Local de Jogo da Rodada'
    p.xaxis.axis_label = 'Pontuação Cedida pelo Adversário no Local de Jogo da Rodada'

    med_x = dataframe_mapa_confronto['pontuacao_cedida'].mean()
    med_y = dataframe_mapa_confronto['pontuacao_executada'].mean()

    med_cedido = Span(location=med_x,
                                    dimension='height', line_color='green',
                                    line_dash='dashed', line_width=3)
    p.add_layout(med_cedido)

    med_exec = Span(location=med_y,
                                    dimension='width', line_color='green',
                                    line_dash='dashed', line_width=3)
    p.add_layout(med_exec)


    q_1 = BoxAnnotation(top=med_y, right=med_x, fill_alpha=0.1, fill_color='red')
    q_2 = BoxAnnotation(bottom=med_y, left = med_x, fill_alpha=0.1, fill_color='green')

    q_3 = BoxAnnotation(top=med_y, left=med_x, fill_alpha=0.1, fill_color='blue')
    q_4 = BoxAnnotation(bottom=med_y, right = med_x, fill_alpha=0.1, fill_color='yellow')

    p.add_layout(q_1)
    p.add_layout(q_2)
    p.add_layout(q_3)
    p.add_layout(q_4)

    p.background_fill_color = 'white'
    p.border_fill_color = None

    return p

# %%

def get_pont_distribution(df_data_scouts___, player_id, dist_user = st.norm, flag_std=4):

    p = figure(title='Distribuição de Probabilidade',
                tools='',
                background_fill_color="#fafafa",
                plot_width=500,
                plot_height=500)

    dist = dist_user

    mean_ = df_data_scouts___[(df_data_scouts___['atleta_id']==player_id) &
                                ((df_data_scouts___['pontuacao']>0))]['pontuacao'].mean()

    std_ = df_data_scouts___[(df_data_scouts___['atleta_id']==player_id) &
                                ((df_data_scouts___['pontuacao']>0))]['pontuacao'].std()

    ponts_player = df_data_scouts___[(df_data_scouts___['atleta_id']==player_id) &
                            ((df_data_scouts___['pontuacao']>0))
                            & (abs(df_data_scouts___['pontuacao']-mean_)<=flag_std*std_)]['pontuacao'].tolist()

    if(len(ponts_player)<3):
        p.add_layout(Title(text=str('Jogador com Qtd Insuficiente de Pontos'), align="center"), "above")
        return p

    depara_clube_logo = df_data_scouts___[['atleta_id','foto','apelido']].drop_duplicates(keep='first')
    logo = depara_clube_logo[depara_clube_logo['atleta_id']==player_id]['foto']
    nome = depara_clube_logo[depara_clube_logo['atleta_id']==player_id]['apelido']

    hist, edges = np.histogram(ponts_player, density=True)
    data = ponts_player
    h1 = max(hist)

    params  = dist.fit(data)
    size=10000
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Build PDF and turn into pandas Series
    x_p = np.linspace(int(min(edges)-1), int(max(edges)+1),size)
    y_p = dist.pdf(x_p, loc=loc, scale=scale, *arg)

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="orange", line_color="white", alpha=0.5)

    p.line(x_p, y_p, line_color="red", line_width=2, alpha=0.7)

    player_pos = df_data_scouts___[(df_data_scouts___['atleta_id']==player_id)]['posicao_nome'].unique().tolist()[0]
    list_ponts = df_data_scouts___[(df_data_scouts___['posicao_nome']==player_pos) &
                        (df_data_scouts___['atleta_id']!=player_id)]['pontuacao'].tolist()

    hist, edges = np.histogram(list_ponts, density=True, bins=50)
    h2 = max(hist)
    # Load data from statsmodels datasets
    data = list_ponts
    params  = dist.fit(data)
    size=10000
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Build PDF and turn into pandas Series
    x = np.linspace(int(min(edges)-1), int(max(edges)+1),size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)


    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="gray", line_color="white", alpha=0.5)

    p.line(x, y, line_color="black", line_width=2, alpha=0.7)
    list_xs = [int(x) for x in range(int(min(edges)-1), int(max(edges)+1),3)]
    source_l = ColumnDataSource(dict(
            url = [logo],
            x_  = [list_xs[-3]],
            y_  = [max(h1,h2)*0.65]
        ))

    image4 = ImageURL(url='url', x='x_', y='y_', anchor="center")
    p.add_glyph(source_l, image4)

            #p.rect(params[0] + 2.5, max(h1,h2), width=30, height=0.025, color="#0d3362", fill_alpha=0.5)
    p.text(list_xs[-3], max(h1,h2)*0.9, nome, x_offset = -30, text_color='#273746', text_font_style="bold")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    p.xaxis.ticker = [int(x) for x in range(int(min(edges)-1), int(max(edges)+1),3)]

    return p
# %%

def unit_poly_verts(r, theta, centre ):
    """Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0= [centre ] * 2
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def radar_patch(r, theta, centre ):
    """ Returns the x and y coordinates corresponding to the magnitudes of
    each variable displayed in the radar plot
    """
    # offset from centre of circle
    offset = 0.01
    yt = (r) * np.sin(theta) + centre
    xt = (r) * np.cos(theta) + centre
    return xt, yt

#%%

def get_radar_plot(cid, pos, df_data_scouts___, df_confrontos_rodada_atual, df_clubes):

    df_execution_team = df_data_scouts___[(df_data_scouts___['posicao_nome'].isin([pos]))].groupby(['clube_name','clube_id']).agg(
        DS = ('DS','sum'),
        FS = ('FS','sum'),
        A = ('A', 'sum'),
        G = ('G', 'sum'),
        F = ('F', 'sum'),
        qj = ('rodada_id',pd.Series.nunique)
    ).reset_index()

    df_execution_team['DS_'] = df_execution_team['DS']/df_execution_team['qj']
    df_execution_team['F_'] = df_execution_team['F']/df_execution_team['qj']
    df_execution_team['A_'] = df_execution_team['A']/df_execution_team['qj']
    df_execution_team['G_'] = df_execution_team['G']/df_execution_team['qj']
    df_execution_team['FS_'] = df_execution_team['FS']/df_execution_team['qj']

    df_execution_team['DS_n'] = (df_execution_team['DS_']/df_execution_team['DS_'].mean() - 1)*100
    df_execution_team['F_n'] = (df_execution_team['F_']/df_execution_team['F_'].mean() - 1)*100
    df_execution_team['FS_n'] = (df_execution_team['FS_']/df_execution_team['FS_'].mean() - 1)*100
    df_execution_team['A_n'] = (df_execution_team['A_']/df_execution_team['A_'].mean() - 1)*100
    df_execution_team['G_n'] = (df_execution_team['G_']/df_execution_team['G_'].mean() - 1)*100

    df_execution_team_res = df_execution_team[['clube_id','clube_name','G_n','DS_n','F_n','FS_n','A_n']]

    df_cedido_team = df_data_scouts___[(df_data_scouts___['posicao_nome'].isin([pos]))].groupby(['opposing_team_id','opposing_team_name']).agg(
        DS = ('DS','sum'),
        FS = ('FS','sum'),
        A = ('A', 'sum'),
        G = ('G', 'sum'),
        F = ('F', 'sum'),
        qj = ('rodada_id',pd.Series.nunique)
    ).reset_index()

    df_cedido_team['DS_'] = df_cedido_team['DS']/df_cedido_team['qj']
    df_cedido_team['F_'] = df_cedido_team['F']/df_cedido_team['qj']
    df_cedido_team['A_'] = df_cedido_team['A']/df_cedido_team['qj']
    df_cedido_team['G_'] = df_cedido_team['G']/df_cedido_team['qj']
    df_cedido_team['FS_'] = df_cedido_team['FS']/df_cedido_team['qj']

    df_cedido_team['DS_n'] = (df_cedido_team['DS_']/df_cedido_team['DS_'].mean() - 1)*100
    df_cedido_team['F_n'] = (df_cedido_team['F_']/df_cedido_team['F_'].mean() - 1)*100
    df_cedido_team['FS_n'] = (df_cedido_team['FS_']/df_cedido_team['FS_'].mean() - 1)*100
    df_cedido_team['A_n'] = (df_cedido_team['A_']/df_cedido_team['A_'].mean() - 1)*100
    df_cedido_team['G_n'] = (df_cedido_team['G_']/df_cedido_team['G_'].mean() - 1)*100

    df_cedido_team_res = df_cedido_team[['opposing_team_id','opposing_team_name','G_n','DS_n','F_n','FS_n','A_n']]

    df_confro_a = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]
    df_confro_b = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]

    df_confro_b['clube_casa_id'] = df_confro_a['clube_visitante_id']
    df_confro_b['clube_visitante_id'] = df_confro_a['clube_casa_id']

    df_confronto_list = df_confro_b.append(df_confro_a)
    df_confronto_list.columns = ['clube_id','opposing_team_id']


    t_n_a = df_confronto_list[df_confronto_list['clube_id']==cid].opposing_team_id.tolist()[0]

    dados_exec = df_execution_team_res[df_execution_team_res['clube_id']==cid].iloc[0,2:]
    dados_ced = df_cedido_team_res[df_cedido_team_res['opposing_team_id']==t_n_a].iloc[0,2:]


    df_escudos = df_clubes[['id','escudo_url']].drop_duplicates()

    esc_ = df_escudos[df_escudos['id']==cid]['escudo_url'].tolist()[0]
    esc_a = df_escudos[df_escudos['id']==t_n_a]['escudo_url'].tolist()[0]

    num_vars = 5

    centre = 0
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    nome = df_execution_team_res[df_execution_team_res['clube_id']==cid]['clube_name'].tolist()[0]
    nome_a = df_cedido_team_res[df_cedido_team_res['opposing_team_id']==t_n_a]['opposing_team_name'].tolist()[0]

    s_ = 600
    wid_ = s_
    hei_ = s_

    p = figure(x_range=Range1d(-260,260),
                            y_range=Range1d(-260,260), width=wid_, height=hei_, tools='')

    p.add_layout(Title(text="Radar - Visão " + nome.upper() + " | " + pos.upper(), align="center"), "above")

    f1 = np.array([x + 100 for x in dados_exec])
    f2 = np.array([x + 100 for x in dados_ced])

    verts = unit_poly_verts(220, theta, centre)
    x = [v[0] for v in verts]
    y = [v[1] for v in verts]
    text = ['G','DS','F','FS','A','']
    source = ColumnDataSource({'x':x + [centre ],'y':y + [200],'text':text})
    labels = LabelSet(x="x",y="y",text="text",source=source,x_offset=-10, y_offset=-10, text_font_style='bold')
    p.add_layout(labels)

    c_ = []
    for x_,y_ in zip(f1,f2):
        xa = x_ - 100
        ya = y_ - 100

        if(xa>=0 and ya>=0):
            c_.append('green')
        elif(xa>=0 and ya<0):
            c_.append('yellow')
        elif(xa<0 and ya>=0):
            c_.append('blue')
        else:
            c_.append('red')


    source_r = ColumnDataSource({'x':x,'y':y,'cc':c_})
    p.circle(x='x', y='y', size=30, fill_color='cc', fill_alpha=0.2, source=source_r, line_color='cc')

    source_imgs = ColumnDataSource({'x':[-200],'y':[200],'escudo_url':[esc_]})
    image2 = ImageURL(url="escudo_url", x="x", y='y', w=90, h =90, anchor="center")
    p.add_glyph(source_imgs, image2)


    source_imgs = ColumnDataSource({'x':[200],'y':[200],'escudo_url':[esc_a]})
    image2 = ImageURL(url="escudo_url", x="x", y='y', w=90, h = 90, anchor="center")
    p.add_glyph(source_imgs, image2)


    verts_ = unit_poly_verts(200, np.linspace(0, 2*np.pi, 100, endpoint=True) + np.pi/2, centre)
    x_ = [v[0] for v in verts_]
    y_ = [v[1] for v in verts_]

    source_ = ColumnDataSource({'x':x_ + [centre ],'y':y_ + [200]})

    p.line(x="x", y="y", source=source_, color='red')

    verts_ = unit_poly_verts(100, np.linspace(0, 2*np.pi, 100, endpoint=True) + np.pi/2, centre)
    x_ = [v[0] for v in verts_]
    y_ = [v[1] for v in verts_]

    source_ = ColumnDataSource({'x':x_ + [centre ],'y':y_ + [100]})

    p.line(x="x", y="y", source=source_, color='grey', line_alpha=1, line_dash='dashed')

    verts_ = unit_poly_verts(50, np.linspace(0, 2*np.pi, 100, endpoint=True) + np.pi/2, centre)
    x_ = [v[0] for v in verts_]
    y_ = [v[1] for v in verts_]

    source_ = ColumnDataSource({'x':x_ + [centre ],'y':y_ + [50]})

    p.line(x="x", y="y", source=source_, color='red', line_alpha=1, line_dash='dashed')

    verts_ = unit_poly_verts(150, np.linspace(0, 2*np.pi, 100, endpoint=True) + np.pi/2, centre)
    x_ = [v[0] for v in verts_]
    y_ = [v[1] for v in verts_]

    source_ = ColumnDataSource({'x':x_ + [centre ],'y':y_ + [150]})

    p.line(x="x", y="y", source=source_, color='green', line_alpha=1, line_dash='dashed')


    source_l = ColumnDataSource({'x':[0]*3,'y':[50,100,150],'text':['-50%', '0', '50%']})
    labels_ = LabelSet(x='x',y='y',text='text',source=source_l,x_offset=-10, y_offset=-20)
    p.add_layout(labels_)

    # example factor:

    #xt = np.array(x)
    flist = [f1,f2]
    colors = ['blue','yellow']
    p_ = []
    for i in range(len(flist)):
        xt, yt = radar_patch(flist[i], theta, centre)
        p_.append(p.patch(x=xt, y=yt, fill_alpha=0.2, fill_color=colors[i]))


    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None

    p.xaxis.ticker = FixedTicker(ticks=[])
    p.yaxis.ticker = FixedTicker(ticks=[])
    p.xaxis.axis_line_color= None
    p.yaxis.axis_line_color= None
    p.outline_line_color = None

    legend = Legend(items=[(fruit, [r]) for (fruit, r) in zip(['Executado - ' + nome.upper(),
                                                                'Cedido - ' + nome_a.upper()], p_)],
                    location=(0.05*wid_, 0*hei_))
    p.add_layout(legend, 'center')

    p.legend.label_text_font_size = '8pt'
    p.legend.border_line_color = None

    return p

# %%
def get_top_players_in_position_in_team(cid, pos, df_clubes, df_data_scouts___, rodada_atual_num):

    de_para_time_logo = df_clubes[['id','escudo_url']].drop_duplicates()

    df_top_scouts = df_data_scouts___[
            (df_data_scouts___['posicao_nome'].isin([pos])) &
            (df_data_scouts___['clube_id']==cid)
                                                ].groupby(['atleta_id','clube_name','clube_id']).agg(
                                                foto = ('foto','first'),
                                                player_name = ('apelido','first'),
                                                scout_ = ('pontuacao','mean'),
                                                qj = ('rodada_id', 'count')
                                            ).reset_index().sort_values(by='scout_',
                                                                        ascending=False)

    df_top_scouts = df_top_scouts[df_top_scouts['qj']>=int(rodada_atual_num*0.5)-1]
    df_top_scouts = df_top_scouts.sort_values(by='scout_', ascending=False).head(10)

    last_game = df_data_scouts___[
            (df_data_scouts___['posicao_nome'].isin([pos])) &
            (df_data_scouts___['clube_id']==cid)
                                                ].sort_values(by='rodada_id').rodada_id.unique().tolist()[-5:]

    last_players = df_data_scouts___[
                (df_data_scouts___['posicao_nome'].isin([pos])) &
                (df_data_scouts___['clube_id']==cid) &
                (df_data_scouts___['rodada_id'].isin(last_game))].groupby(['atleta_id','apelido','clube_name','clube_id']).agg(
                    foto = ('foto','first'),
                qj_r = ('rodada_id', 'count')
            ).reset_index().sort_values(by='qj_r', ascending=False)

    df_top_scouts = pd.merge(df_top_scouts, last_players[['atleta_id','qj_r']], how='left', on='atleta_id').fillna(0)

    max_y_1 = max(df_top_scouts['scout_'].tolist())
    df_top_scouts['label_pos'] = [x + max_y_1/6 for x in df_top_scouts['scout_'].tolist()]

    df_top_scouts['label_pos_data'] = [x*0.9 for x in df_top_scouts['scout_'].tolist()]
    df_top_scouts['label_data'] = ['{0:.1f}'.format(x) for x in df_top_scouts['scout_'].tolist()]

    df_top_scouts['label_pos_data_2'] = [x for x in df_top_scouts['qj'].tolist()]
    df_top_scouts['label_data_2'] = ['{0:.0f}'.format(x) for x in df_top_scouts['qj'].tolist()]

    df_top_scouts['label_pos_data_3'] = [x + max_y_1/2.7 for x in df_top_scouts['scout_'].tolist()]
    df_top_scouts['label_data_3'] = ['{0:.0f}'.format(x) if x>0 else "" for x in df_top_scouts['qj_r'].tolist()]

    df_top_scouts['label_pos_data_4'] = [x + max_y_1/2 for x in df_top_scouts['scout_'].tolist()]
    df_top_scouts['label_data_4'] = ['{0:.0f}'.format(x) if x>0 else "" for x in df_top_scouts['qj'].tolist()]

    df_top_scouts = pd.merge(df_top_scouts,
                                        de_para_time_logo,how='left',left_on='clube_id', right_on='id')

    try:
        df_clube = df_top_scouts.iloc[-2].to_frame().T
    except:
        df_clube = df_top_scouts.iloc[-1].to_frame().T
    df_clube['pos_'] = max_y_1 + max_y_1/3

    source = ColumnDataSource(df_top_scouts)
    f = figure(x_range=df_top_scouts.player_name.tolist(),
                        y_range=Range1d(0,max(df_top_scouts['scout_'].tolist())*1.68),
                    plot_height=600,
                    plot_width = 600)


    b1 = f.vbar(x='player_name', bottom=0, top='scout_', width=0.5, source=source, color='#FFCD58')
    f.hex(x='player_name', y='label_pos_data', size=40, source=source, color='#010100', fill_alpha=0.5)
    f.text(x='player_name', y='label_pos_data', source=source, text='label_data', x_offset=-12, y_offset=+10, text_color='white')

    f.yaxis.axis_label = 'Média de Pontos'

    image2 = ImageURL(url="foto", x="player_name", y='label_pos', w=0.5, h = max_y_1*0.90/3, anchor="center")
    f.add_glyph(source, image2)

    f.circle(x='player_name', y='label_pos_data_3', size=20, source=source, color='#010100', fill_alpha=0.1, legend='# Últimos 5 Jogos')

    f.text(x='player_name', y='label_pos_data_3', source=source, text='label_data_3', x_offset=-5, y_offset=10)

    f.hex(x='player_name', y='label_pos_data_4', size=30, source=source, color='green', fill_alpha=0.1, legend='# Jogos')

    f.text(x='player_name', y='label_pos_data_4', source=source, text='label_data_4', x_offset=-10, y_offset=10)

    f.xaxis.major_label_orientation = math.pi/4

    f.ygrid.grid_line_color = None
    f.xgrid.grid_line_color = None
    f.xaxis.major_label_text_font_size = '12pt'

    nome_time = df_clubes[df_clubes['id']==cid].nome.tolist()[0]
    f.add_layout(Title(text='TOP jogadores em {0} do {1}'.format(pos.upper(), nome_time.upper()), align="center"), "above")

    return f, df_top_scouts.sort_values(by=['qj','scout_'], ascending=False).head(3).atleta_id.unique().tolist()

# %%

def get_round_quantile_player(atleta_id, df_clubes, df_data_scouts___, dataframe_player):

    dataframe_player_original = aux_func.get_datatframe_player_quantile(df_data_scouts___)
    pos_nome = df_data_scouts___[df_data_scouts___['atleta_id']==atleta_id].posicao_nome.tolist()[0]
    dataframe_player_original = dataframe_player_original[dataframe_player_original['player_position']==pos_nome]

    dict_nice_label = dict({
    'chutes no gol': "FD",
     'desarmes executado': "DS",
     'faltas sofridas': "FS",
     'chutes fora': "FF",
     'gols': "G" ,
     'assistencias': "A"
    })

    drug_color = OrderedDict([
        ("quant",   "#0d3362"),
    ])

    df_player_aux = dataframe_player_original[dataframe_player_original['atleta_id']==atleta_id]
 
    resp = []
    for col_ in df_player_aux.columns[-6:]:
        str_ = col_.split("med_")[1].replace("_", " ")
        quantile = stats.percentileofscore(dataframe_player_original[col_], df_player_aux[col_].tolist()[0])
        resp.append([str_, quantile])

    df_quantile = pd.DataFrame(data = resp, columns=['categoria','quant'])
    df_quantile['atleta_id'] = atleta_id
    df_quantile = pd.merge(df_quantile, df_data_scouts___[['atleta_id','foto','apelido']].drop_duplicates(subset=['atleta_id'], keep='first'), how='left')
    df_quantile['color_'] = df_quantile.apply(lambda x: get_color_quantile(x['quant']), axis=1)
    df_quantile['label_cat'] = df_quantile['categoria'].map(dict_nice_label)

    df = df_quantile.copy()
    tid = df_data_scouts___[df_data_scouts___['atleta_id']==atleta_id].clube_id.unique().tolist()[0]

    depara_clube_logo = df_clubes[['id','nome','escudo_url']].drop_duplicates()
    logo = depara_clube_logo[depara_clube_logo['id']==tid]['escudo_url']

    width = 450
    height = 500
    inner_radius = 30
    outer_radius = inner_radius + 12

    minr = sqrt(log(0.1 * 1E4))
    maxr = sqrt(log(1000 * 1E4))
    a = (outer_radius - inner_radius) / (minr - maxr)
    b = inner_radius - a * maxr

    def rad(mic):
        return mic + inner_radius

    big_angle = 2.0 * np.pi / (len(df) + 1)
    small_angle = big_angle / 5
    base_sizing = 50

    p = figure(width=width, height=height, title="",
        x_axis_type=None, y_axis_type=None,
        x_range=(-base_sizing, base_sizing), y_range=(-base_sizing, base_sizing),
        min_border=0, outline_line_color=None,
        background_fill_color='white')

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # annular wedges
    angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
    colors = df_quantile['color_'].tolist()
    p.annular_wedge(
        0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors,
    )

    # small wedges
    p.annular_wedge(0, 0, inner_radius, rad(df.quant/10),
                    -big_angle+angles+2*small_angle, -big_angle+angles+4*small_angle,
                    color=drug_color['quant'])

    #p.annular_wedge(0, 0, inner_radius,rad(df.pc_),
    #                -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
    #                color=drug_color['pc_'])


    # circular axes and lables
    #labels = np.power(10.0, np.arange(1, 2))
    radii = []
    radii.append(inner_radius + 10)
    p.circle(0, 0, radius=inner_radius + 10, fill_color=None, line_color="white")
    p.text(0, inner_radius + 10, ['10'],
           text_font_size="11px", text_align="center", text_baseline="middle")

    path_origin = os.path.dirname(__file__)

    source_ = ColumnDataSource(dict(
        url = ["https://i.ibb.co/fvbCyHB/Imagem1.png"],
        x_  = [0],
        y_  = [inner_radius + 10]
    ))

    image3 = ImageURL(url='url', x='x_', y='y_', w=14, h = 14, anchor="center")
    p.add_glyph(source_, image3)

    source_l = ColumnDataSource(dict(
        url = [logo],
        x_  = [0],
        y_  = [20]
    ))

    image4 = ImageURL(url='url', x='x_', y='y_', w=10, h = 10, anchor="center")
    p.add_glyph(source_l, image4)

    p.text(0, inner_radius, ['Cartologia'],
           text_font_size="20px", text_align="center", text_baseline="middle")



    # radial axes
    p.annular_wedge(0, 0, inner_radius-5, outer_radius+10,
                    -big_angle+angles, -big_angle+angles, color="black")

    # bacteria labels
    #xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))*1.05
    #yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))*1.05
    xr = (inner_radius-14)*np.cos(np.array(-big_angle+angles+3*small_angle))*1
    yr = (inner_radius-14)*np.sin(np.array(-big_angle+angles+3*small_angle))*1
    label_angle=np.array(-big_angle/2+angles)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, df.label_cat, #angle=label_angle,
           text_font_size="14px", text_align="center", text_baseline="middle", text_font_style="bold")
    p.circle(xr, yr, size=23, fill_alpha=0.2, color=colors)

    xr = radii[0]*np.cos(np.array(-big_angle/2 + angles+1*small_angle))*1.3
    yr = radii[0]*np.sin(np.array(-big_angle/2 + angles+1*small_angle))*1.3
    label_angle=np.array(-big_angle/2+angles+1*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side

    source = ColumnDataSource(dict(
        url = [df.foto.unique().tolist()[0]],
        x_  = [0],
        y_  = [4],
        angle_ = label_angle
    ))

    image2 = ImageURL(url='url', x='x_', y='y_', w=16, h = 16, anchor="center")
    p.add_glyph(source, image2)

    p.rect([0], [-8], width=24, height=5, fill_alpha=0.6,
           color="#F2F3F4")

    p.text(0, -8, text=[df_quantile.apelido.unique().tolist()[0]],
           text_font_size="12px", text_align="center", text_baseline="middle")

    # bacteria labels
    xr = (inner_radius-7)*np.cos(np.array(-big_angle+angles+3*small_angle))*1
    yr = (inner_radius-7)*np.sin(np.array(-big_angle+angles+3*small_angle))*1
    label_angle=np.array(-big_angle+angles+3*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, ['{0:.0f}'.format(x) for x in df.quant.tolist()], #angle=label_angle,
           text_font_size="14px", text_align="center", text_baseline="middle", text_font_style="bold")
    p.circle(xr, yr, size=23, fill_alpha=0.2, color="#F2F3F4")

    return p

def get_color_quantile(quant):
    if(quant/100>=0.9):
        return "#1ABC9C"
    elif(quant/100>=0.75):
        return "#D1F2EB"
    elif(quant/100>=0.5):
        return "#FCF3CF"
    elif(quant/100>=0.25):
        return "#FADBD8"
    else:
        return "#CB4335"


def get_datatframe_player_quantile(df_data_scouts___):

    dataframe_player = df_data_scouts___.groupby(['atleta_id','apelido','clube_id','clube_name']).agg(
    player_position = ('posicao_nome', lambda x:x.value_counts().index[0]),
    med_chutes_no_gol = ('FD', 'mean'),
    med_desarmes_executado = ('DS', 'mean'),
    med_faltas_sofridas = ('FS', 'mean'),
    med_chutes_fora = ('FF', 'mean'),
    med_gols = ('G', 'mean'),
    med_assistencias = ('A', 'mean')
    ).reset_index()

    return dataframe_player


#%%

def get_goleiro_plot(df_data_scouts___, df_clubes, rodada_atual_num, legend_desloc = 1.8):

    photo_height = (4.2*rodada_atual_num)*0.2

    depara_clube_logo = df_clubes[['id','nome','escudo_url']].drop_duplicates()

    df_goleiros = df_data_scouts___[(df_data_scouts___['posicao_nome']=='gol')].groupby(['atleta_id','apelido','foto','clube_name']).agg(
                                                            dds = ('DE','sum'),
                                                            gs = ('GS','sum'),
                                                            round_nums = ('rodada_id','count')
                                                ).reset_index().sort_values(by='round_nums', ascending=False).drop_duplicates(subset=['clube_name'], keep='first')



    df_goleiros['dds_gs'] = df_goleiros['dds']/df_goleiros['gs']
    df_goleiros['dds_rn'] = df_goleiros['dds']/df_goleiros['round_nums']

    df_goleiros['dds_gs'] = [x/y if z==0 else i for x,y,z,i in zip(df_goleiros['dds'],
                                                                    df_goleiros['round_nums'],
                                                                    df_goleiros['gs'],
                                                                    df_goleiros['dds_gs'])]

    df_goleiros['label_pos'] = [x+photo_height for x in df_goleiros.dds.tolist()]

    df_goleiros['label_pos_clube'] = [x+2.5*photo_height for x in df_goleiros.dds.tolist()]

    df_goleiros['label_pos_data'] = [x-photo_height/2 for x in df_goleiros.dds.tolist()]

    df_goleiros['label_data'] = ['{0:.0f}'.format(x) for x in df_goleiros['dds'].tolist()]

    df_goleiros = pd.merge(df_goleiros, depara_clube_logo, how='left',left_on='clube_name', right_on='nome')

    df_goleiros['label_pos_data_2'] = [x-legend_desloc for x in df_goleiros.dds_gs.tolist()]

    df_goleiros['label_data_2'] = ['{0:.1f}'.format(x) for x in df_goleiros['dds_gs'].tolist()]

    df_goleiros = df_goleiros.sort_values(by='dds', ascending=False)

    source = ColumnDataSource(df_goleiros)
    f = figure(x_range=df_goleiros.apelido.tolist(),
                y_range=Range1d(0,max(df_goleiros.dds.tolist())*1.8),
                plot_height=500,
                plot_width = 1000)


    f.vbar(x='apelido', bottom=0, top='dds', width=0.6, source=source, color='#FFCD58')
    f.text(x='apelido', y='label_pos_data', source=source, text='label_data', x_offset=-10)
    f.yaxis.axis_label = "Quantidade de Defesas"

    image2 = ImageURL(url="foto", x="apelido", y='label_pos', w=0.85, h = photo_height, anchor="center")
    f.add_glyph(source, image2)

    image2 = ImageURL(url="escudo_url", x="apelido", y='label_pos_clube', w=0.85, h = photo_height, anchor="center")
    f.add_glyph(source, image2)

    # Setting the second y axis range name and range
    f.extra_y_ranges = {"foo": Range1d(start=0, end=max(df_goleiros.dds_gs.tolist())*5)}

    # Adding the second axis to the plot.
    f.add_layout(LinearAxis(y_range_name="foo", axis_label='Média de Defesas por Gol Sofridos'), 'right')

    # Setting the rect glyph params for the second graph.
    # Using the aditional y range named "foo" and "right" y axis here.
    f.line(x='apelido', y='dds_gs',color="green", y_range_name="foo", source=source)

    f.hex(x='apelido', y='dds_gs', size=10, source=source, color='#010100', fill_alpha=0.5, y_range_name="foo")

    f.text(x='apelido', y='label_pos_data_2', source=source, text='label_data_2', x_offset=-10, y_range_name="foo", text_font_size='10pt')

    f.xaxis.major_label_orientation = math.pi/4
    f.xaxis.major_label_text_font_size = '10pt'

    f.ygrid.grid_line_color = None

    return f
# %%

def get_player_dispersion_pos(df_atletas, df_data_scouts___, df_clubes, position, param_size = 0.05, min_num_jogos=2):
    
    pl = df_data_scouts___[(df_data_scouts___['posicao_nome']==position)].groupby('atleta_id')['rodada_id'].count().reset_index()
    pl = pl[pl['rodada_id']>=min_num_jogos]
    pl['atleta_id'] = pl['atleta_id'].astype(int)

    jogadores_provaveis_duvida = df_atletas
    jogadores_provaveis_duvida['atleta_id'] = jogadores_provaveis_duvida['atleta_id'].astype(int)
    pl = pd.merge(pl, jogadores_provaveis_duvida[['atleta_id','status_id']], how='left', on='atleta_id')
    pl = pl[pl['status_id'].isin([2,7])].sort_values(by='status_id', ascending=False)
    pl['atleta_id'] = pl['atleta_id'].astype(str)
    pl = pl['atleta_id'].tolist()

    df_scout = df_data_scouts___[
                       (df_data_scouts___['atleta_id'].isin(pl))].groupby(['atleta_id',
                                    'clube_name']).agg(player_name = ('apelido', 'first'),
                                                        foto = ('foto', 'first'),
                                                          media = ('pontuacao','mean'),
                                                          std = ('pontuacao','std')).reset_index(
                                                            ).fillna(0).sort_values(by='media', ascending=False)

    depara_clube_logo = df_clubes[['id','nome','escudo_url']].drop_duplicates()

    df_scout = pd.merge(df_scout, depara_clube_logo, how='left',left_on='clube_name', right_on='nome')

    sc1 = 'media'
    sc2 = 'std'
    
    df_scout['score'] = df_scout['media'] + df_scout['std']*0.5
    df_scout = df_scout.sort_values(by='score', ascending=False).head(20)
    
    max_x = max(df_scout[sc1].tolist())
    max_y = max(df_scout[sc2].tolist())

    med_x = df_scout[sc1].mean()
    med_y = df_scout[sc2].mean()
    
    df_scout = df_scout.sort_values(by='score', ascending=False).head(20)
    df_scout['clube_pos_x'] = df_scout[sc1] + 0.25*(max_x/11)
    df_scout['clube_pos_y'] = df_scout[sc2] - 0.35*(max_y/11)
    
    p = figure(plot_width=600, plot_height=650, output_backend="webgl",
               x_range = Range1d(min(df_scout[sc1].tolist())*0.9, max(df_scout[sc1].tolist())*1.1),
               y_range = Range1d(min(df_scout[sc2].tolist())*0.9, max(df_scout[sc2].tolist())*1.1))


    source = ColumnDataSource(df_scout)


    p.cross(x=sc1, y=sc2,
             color='navy', fill_alpha=0.1, source=source)

    p.add_layout(Title(text="Dispersão dos Jogadores - Destaque {0}".format(position.upper()), align="center"), "above")

    s_ = min(param_size*(max_x), param_size*(max_y))
    image2 = ImageURL(url="foto",
                      x=sc1,
                      y=sc2,
                      w_units ='screen',
                      h_units = 'screen',
                      w = 50,
                      h = 50,
                      anchor = "center")
    p.add_glyph(source, image2)
    
    image2 = ImageURL(url="escudo_url",
                      x='clube_pos_x',
                      y='clube_pos_y',
                      w_units ='screen',
                      h_units = 'screen',
                      w = 35,
                      h = 35,
                      anchor = "center")
    p.add_glyph(source, image2)

    med_cedido = Span(location=med_x,
                                      dimension='height', line_color='green',
                                      line_dash='dashed', line_width=3)
    p.add_layout(med_cedido)

    med_exec = Span(location=med_y,
                                  dimension='width', line_color='green',
                                  line_dash='dashed', line_width=3)
    p.add_layout(med_exec)
    p.text(x=sc1, y=sc2, source=source, text='player_name', x_offset=-35, y_offset=-20, text_color='black')



    q_1 = BoxAnnotation(top=med_y, right=med_x, fill_alpha=0.1, fill_color='blue')
    q_2 = BoxAnnotation(bottom=med_y, left = med_x, fill_alpha=0.1, fill_color='yellow')

    q_3 = BoxAnnotation(top=med_y, left=med_x, fill_alpha=0.1, fill_color='green')
    q_4 = BoxAnnotation(bottom=med_y, right = med_x, fill_alpha=0.1, fill_color='red')

    p.add_layout(q_1)
    p.add_layout(q_2)
    p.add_layout(q_3)
    p.add_layout(q_4)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.axis_label = "Oscilação na Pontuação do Cartola"
    p.xaxis.axis_label = "Média na Pontuação do Cartola"
    return p


def get_distribution_points_plot(df_data_scouts___, player_id):
    df_player = df_data_scouts___[df_data_scouts___['atleta_id']==player_id]
    foto_url = df_data_scouts___[df_data_scouts___['atleta_id']==player_id].foto.unique().tolist()[0]

    df_agg = df_player.groupby('class_pontuacao').size().reset_index().sort_values(by='class_pontuacao',ascending=True)

    df_expected = pd.DataFrame(columns = ['class_pontuacao'], data = ['Péssimo', 'Ruim', 'Médio', 'Bom', 'Mitou'])

    df_agg = pd.merge(df_expected, df_agg, how='left', on='class_pontuacao').fillna(0)

    dict_labels = dict({
        'Mitou': 'Acima de 10',
        'Bom': 'Entre 5 e 10',
        'Médio': 'Entre 2 e 5',
        'Ruim': 'Entre 0 e 2',
        'Péssimo': 'Abaixo de 0'
    })

    df_agg['label'] = df_agg['class_pontuacao'].map(dict_labels)

    df_agg['perc'] = df_agg[0]*100/df_agg[0].sum()
    df_agg['perc'] = df_agg['perc'].astype(int)
    df_agg['label_text'] = ['{0}%'.format(x) for x in df_agg['perc']]
    df_agg['text_pos'] = df_agg['perc'] + 3

    source = ColumnDataSource({str(c): v.values for c, v in df_agg.items()})

    p = figure(x_range=['Abaixo de 0', 'Entre 0 e 2', 'Entre 2 e 5',
                        'Entre 5 e 10', 'Acima de 10'],
                y_range=Range1d(0,max(df_agg['text_pos'])*1.4),
                plot_height=500,
                plot_width = 500)

    p.add_layout(Title(text="Distribuição de Pontuação", align="center"), "above")

    p.vbar(x='label', top='perc', width=0.7, source=source, color='#E5C29E')

    labels = LabelSet(x='label', y='text_pos', text='label_text',
            x_offset=-10, y_offset=0, source=source, render_mode='canvas', text_font_size="15pt", text_color='#110F1A')

    p.add_layout(labels)
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None


    p.yaxis.minor_tick_line_width = 0
    p.yaxis.major_tick_line_width = 0
    p.yaxis.minor_tick_out = 1

    p.outline_line_color = None
    p.axis.axis_line_color=None
    p.yaxis.major_label_text_font_size = '0pt'
    p.xaxis.major_label_text_font_size = '12pt'

    pad = 3
    for k, class_ in enumerate(df_agg['class_pontuacao'].tolist()):
        list_rodadas = df_data_scouts___[(df_data_scouts___['atleta_id']==player_id) & (df_data_scouts___['class_pontuacao']==class_)].sort_values(by='rodada_id').rodada_id.tolist()
        list_local = df_data_scouts___[(df_data_scouts___['atleta_id']==player_id) & (df_data_scouts___['class_pontuacao']==class_)].sort_values(by='rodada_id').local.tolist()
        n_lements = len(list_rodadas)+1
        h = df_agg[df_agg['class_pontuacao']==class_]['perc'].tolist()[0]
        h_util = h - 2*pad
        incr = h_util/n_lements
        place = 1
        for rod in list_rodadas:

            local_ = list_local[place-1]
            if(local_=='CASA'):
                color_='#725ac1'
            else:
                color_='#242038'


            citation = Label(x=k+0.3, y=(place)*incr, x_units='data', y_units='data',
                            text='Rod {0:.0f}'.format(rod), render_mode='css',
                            text_color= "white",
                            text_font_size= "18px",
                            background_fill_alpha= 0.6,
                                                            background_fill_color= color_,
                                                            border_line_alpha= 0.8,
                                                            border_line_cap= "round",
                                                            border_line_color= color_)

            p.add_layout(citation)

            place = place + 1

    
    citation = Label(x=3, y=max(df_agg['text_pos'])*1.25, x_units='data', y_units='data',
                            text='Rodada em Casa', render_mode='css',
                            text_color= "white",
                            text_font_size= "18px",
                            background_fill_alpha= 0.6,
                                                            background_fill_color= "#725ac1",
                                                            border_line_alpha= 0.8,
                                                            border_line_cap= "round",
                                                            border_line_color= "#725ac1")

    p.add_layout(citation)

    citation = Label(x=3, y=max(df_agg['text_pos'])*1.15, x_units='data', y_units='data',
                            text='Rodada Fora', render_mode='css',
                            text_color= "white",
                            text_font_size= "18px",
                            background_fill_alpha= 0.6,
                                                            background_fill_color= "#242038",
                                                            border_line_alpha= 0.8,
                                                            border_line_cap= "round",
                                                            border_line_color= "#242038")

    p.add_layout(citation)

    source_ = ColumnDataSource(dict(
            url = [foto_url],
            x_  = [0.75],
            y_  = [max(df_agg['text_pos'])*1.2]
        ))

    image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=100, h = 100, anchor="center")
    p.add_glyph(source_, image3)

    return p

def get_confronto_scout_match(df_data_scouts___, df_confrontos_rodada_atual, pos):
    df_execution_team = df_data_scouts___[(df_data_scouts___['posicao_nome'].isin([pos]))].groupby(['clube_name','clube_id']).agg(
            DS = ('DS','sum'),
            FS = ('FS','sum'),
            A = ('A', 'sum'),
            G = ('G', 'sum'),
            F = ('F', 'sum'),
            qj = ('rodada_id',pd.Series.nunique)
        ).reset_index()

    df_execution_team['DS_'] = df_execution_team['DS']/df_execution_team['qj']
    df_execution_team['F_'] = df_execution_team['F']/df_execution_team['qj']
    df_execution_team['A_'] = df_execution_team['A']/df_execution_team['qj']
    df_execution_team['G_'] = df_execution_team['G']/df_execution_team['qj']
    df_execution_team['FS_'] = df_execution_team['FS']/df_execution_team['qj']

    df_execution_team['DS_ne'] = (df_execution_team['DS_']/df_execution_team['DS_'].mean() - 1)*100
    df_execution_team['F_ne'] = (df_execution_team['F_']/df_execution_team['F_'].mean() - 1)*100
    df_execution_team['FS_ne'] = (df_execution_team['FS_']/df_execution_team['FS_'].mean() - 1)*100
    df_execution_team['A_ne'] = (df_execution_team['A_']/df_execution_team['A_'].mean() - 1)*100
    df_execution_team['G_ne'] = (df_execution_team['G_']/df_execution_team['G_'].mean() - 1)*100

    df_execution_team_res = df_execution_team[['clube_id','clube_name','G_ne','DS_ne','F_ne','FS_ne','A_ne']]

    df_cedido_team = df_data_scouts___[(df_data_scouts___['posicao_nome'].isin([pos]))].groupby(['opposing_team_id','opposing_team_name']).agg(
        DS = ('DS','sum'),
        FS = ('FS','sum'),
        A = ('A', 'sum'),
        G = ('G', 'sum'),
        F = ('F', 'sum'),
        qj = ('rodada_id',pd.Series.nunique)
    ).reset_index()

    df_cedido_team['DS_'] = df_cedido_team['DS']/df_cedido_team['qj']
    df_cedido_team['F_'] = df_cedido_team['F']/df_cedido_team['qj']
    df_cedido_team['A_'] = df_cedido_team['A']/df_cedido_team['qj']
    df_cedido_team['G_'] = df_cedido_team['G']/df_cedido_team['qj']
    df_cedido_team['FS_'] = df_cedido_team['FS']/df_cedido_team['qj']

    df_cedido_team['DS_nc'] = (df_cedido_team['DS_']/df_cedido_team['DS_'].mean() - 1)*100
    df_cedido_team['F_nc'] = (df_cedido_team['F_']/df_cedido_team['F_'].mean() - 1)*100
    df_cedido_team['FS_nc'] = (df_cedido_team['FS_']/df_cedido_team['FS_'].mean() - 1)*100
    df_cedido_team['A_nc'] = (df_cedido_team['A_']/df_cedido_team['A_'].mean() - 1)*100
    df_cedido_team['G_nc'] = (df_cedido_team['G_']/df_cedido_team['G_'].mean() - 1)*100

    df_cedido_team_res = df_cedido_team[['opposing_team_id','opposing_team_name','G_nc','DS_nc','F_nc','FS_nc','A_nc']]

    df_confro_a = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]
    df_confro_b = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]

    df_confro_b['clube_casa_id'] = df_confro_a['clube_visitante_id']
    df_confro_b['clube_visitante_id'] = df_confro_a['clube_casa_id']

    df_confronto_list = df_confro_b.append(df_confro_a)
    df_confronto_list.columns = ['clube_id','opposing_team_id']

    base_scout_confronto = pd.merge(pd.merge(df_execution_team_res, df_confronto_list, how='left', on='clube_id'),
    df_cedido_team_res, how='left', on='opposing_team_id')

    for scout in ['G','DS','F','FS','A']:
        base_scout_confronto[f'{scout}_score'] = (base_scout_confronto[f'{scout}_ne']/abs(base_scout_confronto[f'{scout}_ne']) + base_scout_confronto[f'{scout}_nc']/abs(base_scout_confronto[f'{scout}_nc']))/2

    return base_scout_confronto

def get_full_text(jogador_nome, pos_,  ex_):
    msg = f"As oportunidades do {jogador_nome} são em: \n"
    for sc_ in ex_:
        if(sc_[5]=='match_A'):
            msg = msg + f'Em {sc_[0]} o time adversário cede {int(sc_[3]*100)}% mais pontos para {pos_.upper()} do que a média do campeonato e o {jogador_nome} tem uma nota {int(sc_[2])} nesse scout com relação a outros jogadores da mesma posição' + "\n"
        else:
            msg = msg + f'Já em {sc_[0]} apesar do time adversário ceder {int(sc_[3]*100)}% menos pontos para {pos_.upper()} do que a média do campeonato, o {jogador_nome} tem nota {int(sc_[2])} nesse scout com relação a outros jogadores da mesma posição' + "\n"
    return msg

def get_confronto_scout_match_(df_data_scouts___, df_confrontos_rodada_atual, rodada_atual_num, pos, flag_local = False):

    team_id = df_confrontos_rodada_atual.clube_casa_id.tolist() + df_confrontos_rodada_atual.clube_visitante_id.tolist()
    local_de_jogo_da_rodada = len(df_confrontos_rodada_atual)*['CASA'] + len(df_confrontos_rodada_atual)*['FORA']

    dataframe_local_rodada = pd.DataFrame(columns = ['clube_id','local'])
    dataframe_local_rodada['clube_id'] = team_id
    dataframe_local_rodada['local'] = local_de_jogo_da_rodada
    dataframe_local_rodada['flag_mapa'] = 1

    df_data_scouts___adjs = pd.merge(df_data_scouts___, dataframe_local_rodada, how='left', on=['clube_id','local'])
    if(flag_local):
        df_data_scouts___adjs = df_data_scouts___adjs[~pd.isnull(df_data_scouts___adjs['flag_mapa'])]

    df_data_scouts___adjs_ = df_data_scouts___adjs[df_data_scouts___adjs['rodada_id']>=(rodada_atual_num-12)]

    df_execution_team = df_data_scouts___adjs_[(df_data_scouts___['posicao_nome'].isin([pos]))].groupby(['clube_name','clube_id']).agg(
            DS = ('DS','sum'),
            FS = ('FS','sum'),
            A = ('A', 'sum'),
            G = ('G', 'sum'),
            F = ('F', 'sum'),
            qj = ('rodada_id',pd.Series.nunique)
        ).reset_index()

    df_execution_team['DS_'] = df_execution_team['DS']/df_execution_team['qj']
    df_execution_team['F_'] = df_execution_team['F']/df_execution_team['qj']
    df_execution_team['A_'] = df_execution_team['A']/df_execution_team['qj']
    df_execution_team['G_'] = df_execution_team['G']/df_execution_team['qj']
    df_execution_team['FS_'] = df_execution_team['FS']/df_execution_team['qj']

    df_execution_team['DS_ne'] = 1+(df_execution_team['DS_']/df_execution_team['DS_'].mean() - 1)
    df_execution_team['F_ne'] = 1+(df_execution_team['F_']/df_execution_team['F_'].mean() - 1)
    df_execution_team['FS_ne'] = 1+(df_execution_team['FS_']/df_execution_team['FS_'].mean() - 1)
    df_execution_team['A_ne'] = 1+(df_execution_team['A_']/df_execution_team['A_'].mean() - 1)
    df_execution_team['G_ne'] = 1+(df_execution_team['G_']/df_execution_team['G_'].mean() - 1)

    df_execution_team_res = df_execution_team[['clube_id','clube_name','G_ne','DS_ne','F_ne','FS_ne','A_ne']]

    df_cedido_team = df_data_scouts___adjs_[(df_data_scouts___['posicao_nome'].isin([pos]))].groupby(['opposing_team_id','opposing_team_name']).agg(
        DS = ('DS','sum'),
        FS = ('FS','sum'),
        A = ('A', 'sum'),
        G = ('G', 'sum'),
        F = ('F', 'sum'),
        qj = ('rodada_id',pd.Series.nunique)
    ).reset_index()

    df_cedido_team['DS_'] = df_cedido_team['DS']/df_cedido_team['qj']
    df_cedido_team['F_'] = df_cedido_team['F']/df_cedido_team['qj']
    df_cedido_team['A_'] = df_cedido_team['A']/df_cedido_team['qj']
    df_cedido_team['G_'] = df_cedido_team['G']/df_cedido_team['qj']
    df_cedido_team['FS_'] = df_cedido_team['FS']/df_cedido_team['qj']

    df_cedido_team['DS_nc'] = 1+(df_cedido_team['DS_']/df_cedido_team['DS_'].mean() - 1)
    df_cedido_team['F_nc'] = 1+(df_cedido_team['F_']/df_cedido_team['F_'].mean() - 1)
    df_cedido_team['FS_nc'] = 1+(df_cedido_team['FS_']/df_cedido_team['FS_'].mean() - 1)
    df_cedido_team['A_nc'] = 1+(df_cedido_team['A_']/df_cedido_team['A_'].mean() - 1)
    df_cedido_team['G_nc'] = 1+(df_cedido_team['G_']/df_cedido_team['G_'].mean() - 1)

    df_cedido_team_res = df_cedido_team[['opposing_team_id','opposing_team_name','G_nc','DS_nc','F_nc','FS_nc','A_nc']]

    df_confro_a = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]
    df_confro_b = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]

    df_confro_b['clube_casa_id'] = df_confro_a['clube_visitante_id']
    df_confro_b['clube_visitante_id'] = df_confro_a['clube_casa_id']

    df_confronto_list = df_confro_b.append(df_confro_a)
    df_confronto_list.columns = ['clube_id','opposing_team_id']

    base_scout_confronto = pd.merge(pd.merge(df_execution_team_res, df_confronto_list, how='left', on='clube_id'),
    df_cedido_team_res, how='left', on='opposing_team_id')

    for scout in ['G','DS','F','FS','A']:
        base_scout_confronto[f'{scout}_score'] = (base_scout_confronto[f'{scout}_ne']*base_scout_confronto[f'{scout}_nc'])

    return base_scout_confronto

def dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___,
 df_confrontos_rodada_atual, df_clubes, pos_, alpha_risk=1, beta_risk=0.25, min_num_jogos = 0):

    dataframe_player = df_data_scouts___.groupby(['atleta_id','apelido','clube_id','clube_name']).agg(
    qtd_rods = ('rodada_id', pd.Series.nunique),
    player_position = ('posicao_nome', lambda x:x.value_counts().index[0]),
    med_chutes_no_gol = ('FD', 'mean'),
    med_desarmes_executado = ('DS', 'mean'),
    med_faltas_sofridas = ('FS', 'mean'),
    med_chutes_fora = ('FF', 'mean'),
    med_gols = ('G', 'mean'),
    med_assistencias = ('A', 'mean')
    ).reset_index()

    dataframe_player = dataframe_player[dataframe_player['qtd_rods']>=min_num_jogos]
    dataframe_player = dataframe_player[(dataframe_player['med_chutes_no_gol'] + dataframe_player['med_desarmes_executado'] + dataframe_player['med_faltas_sofridas'] + dataframe_player['med_chutes_fora'] + dataframe_player['med_gols'] + dataframe_player['med_assistencias']) > 0]

    base_scout_confronto = aux_func.get_confronto_scout_match_(df_data_scouts___, df_confrontos_rodada_atual, rodada_atual_num, pos_)

    df_mapa = aux_func.get_dataframe_mapa_confronto(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, pos_, rodada_atual_num)
    df_mapa['score_conquista'] = (df_mapa['pontuacao_executada']-df_mapa['pontuacao_executada'].mean())/df_mapa['pontuacao_executada'].mean()
    df_mapa['score_cedido'] = (df_mapa['pontuacao_cedida']-df_mapa['pontuacao_cedida'].mean())/df_mapa['pontuacao_cedida'].mean()

    df_mapa['score_final'] = (1-alpha_risk)*df_mapa['score_conquista'] + alpha_risk*df_mapa['score_cedido']
    df_mapa = df_mapa.sort_values(by='score_final', ascending=False)

    dict_nice_label = dict({
        'chutes no gol': "FD",
        'desarmes executado': "DS",
        'faltas sofridas': "FS",
        'chutes fora': "FF",
        'gols': "G" ,
        'assistencias': "A"
        })

    stats_ = df_data_scouts___.groupby(['atleta_id']).agg(
        med_ = ('pontuacao','mean'),
        std_ = ('pontuacao','std'),
    ).reset_index()
    stats_['pond_'] = [max(1-x/y,beta_risk) if y!=0 else 0 for x,y in zip(stats_['std_'], stats_['med_'])]

    for scout_ in list(dict_nice_label.keys()):
        scout__ = 'med_' + scout_.replace(" ","_")
        dataframe_player[dict_nice_label[scout_]] = [stats.percentileofscore(dataframe_player[scout__], x) for x in dataframe_player[scout__]]

    #df_atletas['atleta_id'] = df_atletas['atleta_id'].astype(str)
    #dataframe_player['atleta_id'] = dataframe_player['atleta_id'].astype(str)
    dataframe_player_at_pos = dataframe_player[(dataframe_player['player_position']==pos_) &
    (dataframe_player['atleta_id'].isin(df_atletas[df_atletas['status_id']==7].atleta_id.tolist()))]

    dataframe_player_at_pos['F'] = (dataframe_player_at_pos['FF'] + dataframe_player_at_pos['FD'])/2

    base_final_ = pd.DataFrame(columns=['atleta_id', 'apelido', 'clube_id', 'clube_name', 'player_position',
        'med_chutes_no_gol', 'med_desarmes_executado', 'med_faltas_sofridas',
        'med_chutes_fora', 'med_gols', 'med_assistencias', 'FD', 'DS', 'FS',
        'FF', 'G', 'A', 'F', 'score_log', 'team_log'])

    for cid_ in df_mapa.clube_id.tolist():

        list_iters_ = ['G_score', 'DS_score', 'F_score', 'FS_score',
            'A_score']
        list_iters_p = ['G', 'DS', 'F', 'FS', 'A']

        if(pos_=='ata'):
            list_weight_ = [12, 1, 4, 3, 7.5]
        elif(pos_=='mei'):
            list_weight_ = [10, 1, 2, 2, 10]
        elif(pos_=='zag'):
            list_weight_ = [3, 6, 2, 4, 3]
        elif(pos_=='lat'):
            list_weight_ = [4, 6, 1, 2, 10]

        df_team_opts = dataframe_player_at_pos[dataframe_player_at_pos['clube_id']==cid_]
        df_demanda = base_scout_confronto[base_scout_confronto['clube_id']==cid_]
        team_log = []
        score_log = []
        for i_ in range(0,len(df_team_opts)):
            match_player_score = 0
            player_log = []
            for x,y,w_ in zip(list_iters_, list_iters_p,list_weight_):
                match_ = df_demanda[x].tolist()[0]
                player = df_team_opts[y].tolist()[i_]
                #op_team_ = df_demanda[f'{y}_score'].tolist()[0]
                op_team_ = (1-alpha_risk)*df_demanda[f'{y}_ne'].tolist()[0] + alpha_risk*df_demanda[f'{y}_nc'].tolist()[0]
                #if((match_>=1) and (player>50)):
                #    match_player_score = match_player_score + w_
                #    player_log.append([y, match_, player, op_team_, player/100, 'match_A'])
                #elif((match_==0) and (player>50)):
                match_player_score = match_player_score + w_*op_team_*player/100
                player_log.append([y, match_, player, op_team_, player/100,'match_single'])

            team_log.append(player_log)
            score_log.append(match_player_score)

        df_team_opts['score_log'] = score_log
        df_team_opts['team_log'] = team_log

        base_final_ = base_final_.append(df_team_opts)
    base_final_ = base_final_.sort_values(by='score_log', ascending=False)
    base_final__ = pd.merge(base_final_,stats_, how='left', on='atleta_id')
    base_final__['score_pond'] = base_final__['pond_']*base_final__['score_log']
    base_final__ = base_final__.sort_values(by='score_pond', ascending=False)
    base_final__['full_text'] = [aux_func.get_full_text(jogador_nome, pos_, ex_) for jogador_nome, ex_ in zip(base_final__['apelido'], base_final__['team_log'])]
    
    return base_final__

def draw_pitch(width = 700, height = 500,
                measure = 'metres',
                fill_color = '#B3DE69', fill_alpha = 0.5,
                line_color = 'grey', line_alpha = 1,
                hspan = [-52.5, 52.5], vspan = [-34, 34],
                arcs = True):
    '''
    -----
    Draws and returns a pitch on a Bokeh figure object with width 105m and height 68m
    p = drawpitch()
    -----
    If you are using StatsBomb Data with a 120x80yard pitch, use:
    measure = 'SB'
    -----
    If you are using Opta Data, use:
    measure = 'Opta'
    -----
    If you are using any other pitch size, set measure to yards or metres
    for correct pitch markings and
    hspan = [left, right] // eg. for SBData this is: hspan = [0, 120]
    vspan = [bottom, top] //
    to adjust the plot to your needs.
    -----
    set arcs = False to not draw the penaltybox arcs
    '''

    # measures:
    # goalcenter to post, fiveyard-box-length, fiveyard-width,
    # box-width, penalty-spot x-distance, circle-radius


    if measure == 'yards':
        measures = [4, 6, 10, 18, 42, 12, 10]
    elif (measure == 'SBData')|(measure == 'StatsBomb')|(measure == 'statsbomb')|(measure == 'SB'):
        measures = [4, 6, 10, 18, 44, 12, 10]
        hspan = [0, 120]
        vspan = [0, 80]
    elif measure == 'Opta':
        measures = [4.8, 5.8, 13.2, 17, 57.8, 11.5, 8.71]
        hspan = [0, 100]
        vspan = [0, 100]
    else: #if measure = metres or whatever else
        measures = [3.66, 5.5, 9.16, 16.5, 40.32, 11, 9.15]

    hmid = (hspan[1]+hspan[0])/2
    vmid = (vspan[1]+vspan[0])/2

    p = figure(width = width,
        height = height,
        x_range = Range1d(hspan[0], hspan[1]),
        y_range = Range1d(vspan[0], vspan[1]),
        tools = [])

    boxes = p.quad(top = [vspan[1], vmid+measures[2], vmid+measures[4]/2, vmid+measures[4]/2, vmid+measures[2]],
           bottom = [vspan[0], vmid-measures[2], vmid-measures[4]/2, vmid-measures[4]/2, vmid-measures[2]],
           left = [hspan[0], hspan[1]-measures[1], hspan[1]-measures[3], hspan[0]+measures[3], hspan[0]+measures[1]],
           right = [hspan[1], hspan[1], hspan[1], hspan[0], hspan[0]],
           color = fill_color,
           alpha = [fill_alpha,0,0,0,0], line_width = 2,
           line_alpha = line_alpha,
           line_color = line_color)
    boxes.selection_glyph = boxes.glyph
    boxes.nonselection_glyph = boxes.glyph

    #middle circle
    p.circle(x=[hmid], y=[vmid], radius = measures[6],
            color = line_color,
            line_width = 2,
            fill_alpha = 0,
            fill_color = 'grey',
            line_color= line_color)

    if arcs == True:
        p.arc(x=[hspan[0]+measures[5], hspan[1]-measures[5]], y=[vmid, vmid],
            radius = measures[6],
            start_angle = [(2*pi-np.arccos((measures[3]-measures[5])/measures[6])), pi - np.arccos((measures[3]-measures[5])/measures[6])],
            end_angle = [np.arccos((measures[3]-measures[5])/measures[6]), pi + np.arccos((measures[3]-measures[5])/measures[6])],
            color = line_color,
            line_width = 2)

    p.circle([hmid, hspan[1]-measures[5], hspan[0]+measures[5]], [vmid, vmid, vmid], size=5, color=line_color, alpha=1)
    #midfield line
    p.line([hmid, hmid], [vspan[0], vspan[1]], line_width = 2, color = line_color)
    #goal lines
    p.line((hspan[1],hspan[1]),(vmid+measures[0],vmid-measures[0]), line_width = 6, color = 'white')
    p.line((hspan[0],hspan[0]),(vmid+measures[0],vmid-measures[0]), line_width = 6, color = 'white')
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False

    return p

def get_pitch_top_player(df_atletas,
                         df_data_scouts___,
                         df_confrontos_rodada_atual,
                         df_clubes,
                         alpha_risk_,
                         rodada_atual_num,
                         ata_class = 'seguro',
                         mei_class = 'seguro',
                         lat_class = 'ousado',
                         zag_class = 'moderado',
                         gol_class = 'moderado', min_num_jogos_ = 0):
    
    dict_place = dict({
        'delta_escudo':{
            'delta_x':-6,
            'delta_y':2
        },
        'delta_label':{
            'delta_x':-3,
            'delta_y':-9
        },
        'ata_centro': {
            'x':40,
            'y':5
        },
        'ata_ponta_esquerda': {
            'x':30,
            'y':25
        },
        'ata_ponta_direita': {
            'x':30,
            'y':-15
        },
        'mei_1':{
            'x': 5,
            'y': 25
        },
        'mei_2':{
            'x': 5,
            'y': 5
        },
        'mei_3':{
            'x': 5,
            'y': -15
        },
        'lat_1':{
            'x': -15,
            'y': -20
        },
        'lat_2':{
            'x': -15,
            'y': 25
        },
        'zag_1':{
            'x': -25,
            'y': 10
        },
        'zag_2':{
            'x': -25,
            'y': -5
        },
        'gol': {
            'x':-46,
            'y': 0
        }
    })

    ata_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'ata', alpha_risk=alpha_risk_, beta_risk=0.25, min_num_jogos = min_num_jogos_)
    lat_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'lat', alpha_risk=alpha_risk_, beta_risk=0.25, min_num_jogos = min_num_jogos_)
    zag_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'zag', alpha_risk=alpha_risk_, beta_risk=0.25, min_num_jogos =  min_num_jogos_)
    mei_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'mei', alpha_risk=alpha_risk_, beta_risk=0.25, min_num_jogos = min_num_jogos_)
    gol_ = aux_func.get_table_goalkeepers(df_data_scouts___, df_confrontos_rodada_atual, df_atletas, gamma_risk=alpha_risk_)

    list_ata = []
    if(ata_class=='seguro'):
        list_ata = ata_.atleta_id.tolist()[:3]
    elif(ata_class=='moderado'):
        list_ata = ata_.atleta_id.tolist()[3:6]
    elif(ata_class=='ousado'):
        list_ata = ata_.atleta_id.tolist()[6:9]

    list_mei = []
    if(mei_class=='seguro'):
        list_mei = mei_.atleta_id.tolist()[:3]
    elif(mei_class=='moderado'):
        list_mei = mei_.atleta_id.tolist()[3:6]
    elif(mei_class=='ousado'):
        list_mei = mei_.atleta_id.tolist()[6:9]

    list_lat = []
    if(lat_class=='seguro'):
        list_lat = lat_.atleta_id.tolist()[:2]
    elif(lat_class=='moderado'):
        list_lat = lat_.atleta_id.tolist()[2:4]
    elif(lat_class=='ousado'):
        list_lat = lat_.atleta_id.tolist()[4:6]

    list_zag = []
    if(zag_class=='seguro'):
        list_zag = zag_.atleta_id.tolist()[:2]
    elif(zag_class=='moderado'):
        list_zag = zag_.atleta_id.tolist()[2:4]
    elif(zag_class=='ousado'):
        list_zag = zag_.atleta_id.tolist()[4:6]

    list_gol = []
    if(gol_class=='seguro'):
        list_gol = gol_.atleta_id.tolist()[0]
    elif(gol_class=='moderado'):
        list_gol = gol_.atleta_id.tolist()[1]
    elif(gol_class=='ousado'):
        list_gol = gol_.atleta_id.tolist()[2]

    list_ = list_ata + list_mei + list_lat + list_zag + [list_gol]


    p = aux_func.draw_pitch(width = 700, height = 500,
                    measure = 'metres',
                    fill_color = '#B3DE69', fill_alpha = 0.5,
                    line_color = 'grey', line_alpha = 1,
                    hspan = [-52.5, 52.5], vspan = [-34, 34],
                    arcs = True)

    for player_, dict_p in zip(list_, list(dict_place.keys())[2:]):
        url_ = df_data_scouts___[df_data_scouts___['atleta_id']==player_].foto.tolist()[0]  
        clube_name = df_data_scouts___[df_data_scouts___['atleta_id']==player_].clube_name.tolist()[0]
        apelido = df_data_scouts___[df_data_scouts___['atleta_id']==player_].apelido.tolist()[0]
        escudo_ = df_clubes[df_clubes['nome']==clube_name].escudo_url.tolist()[0]

        source_ = ColumnDataSource(dict(
                url = [url_],
                x_  = [dict_place[dict_p]['x']],
                y_  = [dict_place[dict_p]['y']]
            ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
        p.add_glyph(source_, image3)

        source_ = ColumnDataSource(dict(
                url = [escudo_],
                x_  = [dict_place[dict_p]['x'] + dict_place['delta_escudo']['delta_x']],
                y_  = [dict_place[dict_p]['y'] + dict_place['delta_escudo']['delta_y']]
            ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=30, h = 30, anchor="center")
        p.add_glyph(source_, image3)

        #citation = Label(x=dict_place[dict_p]['x'] + dict_place['delta_label']['delta_x'],
        #                y=dict_place[dict_p]['y'] + dict_place['delta_label']['delta_y'],
        #                x_units='data', y_units='data',
        #                            text=apelido, render_mode='css',
        #                            text_color= "black",
        #                            text_font_size= "14px",
        #                            background_fill_alpha= 0.6,
        #                                                            background_fill_color= 'white',
        #                                                            border_line_alpha= 0.8,
        #                                                            border_line_cap= "round",
        #                                                            border_line_color= 'white')

        #p.add_layout(citation)

        source = ColumnDataSource(dict(x=[dict_place[dict_p]['x'] + dict_place['delta_label']['delta_x']+6.25*len(apelido)/11.5],
         y=[dict_place[dict_p]['y'] + dict_place['delta_label']['delta_y']+1.25], w=[16.5*len(apelido)/11.5], h=[2.5]))


        glyph = Rect(x="x", y="y", width="w", height="h", fill_color="#8EFC6E", fill_alpha=0.6, line_alpha=0)
        p.add_glyph(source, glyph)

        source = ColumnDataSource(dict(x=[dict_place[dict_p]['x'] + dict_place['delta_label']['delta_x']],
        y=[dict_place[dict_p]['y'] + dict_place['delta_label']['delta_y']], text=[apelido]))

        glyph = Text(x="x", y="y", text="text",
                                    text_color= "black",
                                    text_font_size= "14px")

        p.add_glyph(source, glyph)

    return p, list_

def get_table_goalkeepers(df_data_scouts___, df_confrontos_rodada_atual, df_atletas, gamma_risk=1):
    finalizacoes_df = df_data_scouts___.groupby(['clube_id','clube_name'])[['FF','FD','G','FT']].sum().reset_index()
    finalizacoes_df['total'] = finalizacoes_df['FF'] + finalizacoes_df['FD'] + finalizacoes_df['G'] + finalizacoes_df['FT']
    finalizacoes_df['acc_no_gol'] = (finalizacoes_df['FD'])/finalizacoes_df['total']
    finalizacoes_df['acc_em_gol'] = (finalizacoes_df['G'])/finalizacoes_df['total']
    finalizacoes_df['score_'] = finalizacoes_df['acc_no_gol']/finalizacoes_df['acc_em_gol']
    finalizacoes_df = finalizacoes_df.sort_values(by='score_', ascending=False)


    goleiros_df = df_data_scouts___.groupby(['clube_id','clube_name'])[['DE','GS']].sum().reset_index()
    goleiros_df['score_'] = goleiros_df['DE']/goleiros_df['GS']
    goleiros_df = goleiros_df.sort_values(by='score_', ascending=False)

    df_confro_a = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]
    df_confro_b = df_confrontos_rodada_atual[['clube_casa_id','clube_visitante_id']]

    df_confro_b['clube_casa_id'] = df_confro_a['clube_visitante_id']
    df_confro_b['clube_visitante_id'] = df_confro_a['clube_casa_id']

    df_confronto_list = df_confro_b.append(df_confro_a)
    df_confronto_list.columns = ['clube_id','opposing_team_id']

    merge_1 = pd.merge(goleiros_df, df_confronto_list, how='left', on='clube_id')
    merge_2 = pd.merge(merge_1, finalizacoes_df, how='left', left_on='opposing_team_id', right_on='clube_id')
    merge_2.columns = ['clube_id', 'clube_name', 'DE', 'GS', 'score_goleiro',
        'opposing_team_id', 'XX', 'opposing_team_name', 'FF', 'FD', 'G', 'FT',
        'total', 'acc_no_gol', 'acc_em_gol', 'score_adv']

    tabela_escal_goleiros = merge_2.drop(['XX'], axis=1)

    tabela_escal_goleiros['score_goleiro'] = tabela_escal_goleiros['score_goleiro']/tabela_escal_goleiros['score_goleiro'].max()
    tabela_escal_goleiros['score_adv'] = tabela_escal_goleiros['score_adv']/tabela_escal_goleiros['score_adv'].max()
    tabela_escal_goleiros['score_final'] = tabela_escal_goleiros['score_goleiro'] + gamma_risk*tabela_escal_goleiros['score_adv']
    tabela_escal_goleiros = tabela_escal_goleiros.sort_values(by='score_final', ascending=False)

    gols_rodada = df_atletas[(df_atletas['status_id']==7) & (df_atletas['posicao_id']==1)][['clube_id','atleta_id','apelido']]

    tabela_escal_goleiros = pd.merge(tabela_escal_goleiros, gols_rodada, how='left', on='clube_id')
    tabela_escal_goleiros = tabela_escal_goleiros[~pd.isnull(tabela_escal_goleiros['atleta_id'])]
    return tabela_escal_goleiros.head(5)
# %%

def get_pont_avg_evol_(atleta_id_p_, df_data_scouts___, base_global_partidas, df_clubes):
    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==atleta_id_p_].posicao_nome.tolist()[0]
    clube_id_ = df_data_scouts___[df_data_scouts___['atleta_id']==atleta_id_p_].clube_id.tolist()[0]
    rodada_ = []
    rodada_jog = []
    for i_ in df_data_scouts___.sort_values(by='rodada_id').rodada_id.unique().tolist():
        base_pos = df_data_scouts___[(df_data_scouts___['rodada_id']<=i_) &
        (df_data_scouts___['posicao_nome']==pos_)].groupby(['atleta_id','apelido'])['pontuacao'].mean().reset_index().sort_values(by='pontuacao', ascending=False)

        media_top10 = base_pos.head(10).pontuacao.mean()
        max_media = base_pos.pontuacao.max()
        media_last30 = base_pos.tail(int(len(base_pos)/2)).pontuacao.mean()
        min_media = base_pos.pontuacao.min()

        base_jogador_ = df_data_scouts___[(df_data_scouts___['rodada_id']<=i_) & (df_data_scouts___['atleta_id']==atleta_id_p_)]
        media_jogador = None
        if(len(base_jogador_[base_jogador_['rodada_id']==i_])>0):

            media_jogador = base_jogador_.pontuacao.mean()
            pontuacao_na_rodada = base_jogador_[base_jogador_['rodada_id']==i_].pontuacao.tolist()[0]

            base_partidas = base_global_partidas[base_global_partidas['rodada_id']==i_]
            if(clube_id_ in base_partidas.clube_casa_id.tolist()):
                op_ = base_partidas[base_partidas['clube_casa_id']==clube_id_].clube_visitante_id.tolist()[0]
                local_ = 'casa'
            else:
                op_ = base_partidas[base_partidas['clube_visitante_id']==clube_id_].clube_casa_id.tolist()[0]
                local_ = 'fora'

            op_nome = df_clubes[df_clubes['id']==op_].nome.tolist()[0]
            op_escudo = df_clubes[df_clubes['id']==op_].escudo_url.tolist()[0]
            rodada_jog.append([i_, pontuacao_na_rodada, media_jogador, op_, local_, op_nome, op_escudo])

        rodada_.append([i_, media_jogador, media_top10, media_last30, max_media, min_media])

    tabela_resposta = pd.DataFrame(data = rodada_, columns = ['rodada_id', 'media_jogador', 'media_top10', 'media_last30', 'max_media', 'min_media'])
    tabela_resposta = tabela_resposta.fillna(method='ffill').fillna(0)


    tabela_jogador_resposta = pd.DataFrame(data = rodada_jog, columns = ['i_', 'pontuacao_na_rodada', 'media_jogador', 'op_', 'local_', 'op_nome', 'op_escudo'])

    tabela_resposta['topo_'] = tabela_resposta.max_media.max()*1.2
    tabela_resposta['bottom_'] = tabela_resposta.min_media.min()-1
    tabela_resposta = tabela_resposta.tail(12)

    source = ColumnDataSource(tabela_resposta)
    tabela_jogador_resposta['pont_label'] = ['({0:.1f})'.format(x) for x in tabela_jogador_resposta['pontuacao_na_rodada']]
    tabela_jogador_resposta['media_label'] = ['M:{0:.1f}'.format(x) for x in tabela_jogador_resposta['media_jogador']]

    tabela_jogador_resposta = tabela_jogador_resposta.tail(12)
    source_jogador = ColumnDataSource(tabela_jogador_resposta)


    TOOLS = "pan,wheel_zoom,box_zoom,reset"
    p = figure(tools=TOOLS, width=800, height=600)
    p.y_range = Range1d(tabela_resposta.min_media.min()-1, min(tabela_resposta.max_media.max()*1.2, 22))

    p.x_range = Range1d(tabela_resposta.rodada_id.min(), tabela_resposta.rodada_id.max()+0.35)

    p.line(x='rodada_id', y='media_jogador', source=source, line_width=3)

    band = Band(base='rodada_id', lower='media_last30', upper='media_top10', source=source, level='underlay',
                fill_alpha=0.7, line_width=1, line_color='black')
    p.add_layout(band)


    band = Band(base='rodada_id', lower='media_top10', upper='topo_', source=source, level='underlay', fill_alpha=0.2, fill_color='#BEF5AD',
                line_width=1, line_color='black')
    p.add_layout(band)

    band = Band(base='rodada_id', lower='bottom_', upper='media_last30', source=source, level='underlay', fill_alpha=0.2, fill_color='#F7BBB0',
                line_width=1, line_color='black')
    p.add_layout(band)

    p.scatter(x='rodada_id', y='max_media', line_color='#3BF516', fill_alpha=0.9, size=5, source=source, fill_color='#3BF516')
    p.line(x='rodada_id', y='max_media', source=source, line_color='#3BF516', line_alpha=0.6)

    p.text(source=source_jogador, x='i_', y='media_jogador', text='pont_label', y_offset=-10, x_offset=-20)

    p.text(source=source_jogador, x='i_', y='media_jogador', text='media_label', y_offset=-35, x_offset=-20)

    p.scatter(x='rodada_id', y='min_media', line_color='#F51616', fill_alpha=0.9, size=5, source=source, fill_color='#F51616')
    p.line(x='rodada_id', y='min_media', source=source, line_color='#F51616', line_alpha=0.6)

    image3 = ImageURL(url='op_escudo', x='i_', y='media_jogador', w_units='screen', h_units='screen', w=40, h = 40, anchor="top")
    p.add_glyph(source_jogador, image3)


    foto_ = df_data_scouts___[df_data_scouts___['atleta_id']==atleta_id_p_].foto.tolist()[0]
    source_ = ColumnDataSource(dict(
                    url = [foto_],
                    x_  = [tabela_resposta.rodada_id.max()-1],
                    y_  = [tabela_resposta.max_media.max()-2]
                ))

    image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=100, h = 100, anchor="center")
    p.add_glyph(source_, image3)

    p.ygrid[0].grid_line_color=None
    p.xaxis.axis_label = 'Rodada'
    p.yaxis.axis_label = 'Média de Pontos'
    p.yaxis.minor_tick_out = 0
    p.xaxis.minor_tick_out = 0

    return p

#%%

def get_resistance_plot(op_, df_data_scouts___, base_global_partidas, df_clubes, flag_relative = True, flag_insta = True):
    
    df_data_scouts___cp = df_data_scouts___.copy()
    df_data_scouts___cp['pontuacao_corr'] = df_data_scouts___cp['pontuacao'] - 5*df_data_scouts___cp['SG']

    if(flag_relative):
        base_pontuacao_conquistada = df_data_scouts___cp.groupby(['rodada_id',
        'clube_id','clube_name','posicao_nome'])['pontuacao_corr'].sum().reset_index()
    else:
        base_pontuacao_conquistada = df_data_scouts___cp.groupby(['rodada_id',
        'clube_id','clube_name','posicao_nome'])['pontuacao_corr'].mean().reset_index()


    depara_partidas = base_global_partidas[~pd.isnull(base_global_partidas['placar_oficial_mandante'])][['clube_casa_id','clube_visitante_id','rodada_id']]
    depara_partidas['local'] = 'CASA'
    depara_partidas_cp = depara_partidas.copy()
    depara_partidas_cp['clube_casa_id'] = depara_partidas['clube_visitante_id']
    depara_partidas_cp['clube_visitante_id'] = depara_partidas['clube_casa_id']
    depara_partidas_cp['local'] = 'FORA'

    base_partidas = depara_partidas.append(depara_partidas_cp)
    base_partidas.columns = ['clube_id','opponent_id','rodada_id','local']

    base_pontuacao_conquistada_w_partidas = pd.merge(base_pontuacao_conquistada,
    base_partidas, how='left', on=['clube_id','rodada_id'])

    tabela_agg_delta_agg_pos = pd.DataFrame(columns = ['local_op','delta','posicao_nome'])

    for pos_ in df_data_scouts___.posicao_nome.unique().tolist():

        base_times_versus_time_analisado = base_pontuacao_conquistada_w_partidas[(base_pontuacao_conquistada_w_partidas['opponent_id']==op_) &
        (base_pontuacao_conquistada_w_partidas['posicao_nome']==pos_) & (base_pontuacao_conquistada_w_partidas['rodada_id']>1)]

        vec_medias = []
        for time_, rid_ in zip(base_times_versus_time_analisado['clube_id'], base_times_versus_time_analisado['rodada_id']):
            if(flag_relative):
                media_ate_rid_ = df_data_scouts___cp[(df_data_scouts___cp['clube_id']==time_) &
                                                (df_data_scouts___cp['posicao_nome']==pos_) &
                                                (df_data_scouts___cp['rodada_id']<rid_)].groupby(['rodada_id'])['pontuacao_corr'].sum().reset_index().pontuacao_corr.mean()
            else:
                media_ate_rid_ = df_data_scouts___cp[(df_data_scouts___cp['clube_id']==time_) &
                                                (df_data_scouts___cp['posicao_nome']==pos_) &
                                                (df_data_scouts___cp['rodada_id']<rid_)].groupby(['rodada_id'])['pontuacao_corr'].mean().reset_index().pontuacao_corr.mean()
            vec_medias.append(media_ate_rid_)

        base_times_versus_time_analisado['media_until'] = vec_medias

        if(flag_relative):
            base_times_versus_time_analisado['delta'] = [(x-y)/abs(y) if y!=0 else 0 for x,y in zip(base_times_versus_time_analisado['pontuacao_corr'],base_times_versus_time_analisado['media_until'])]
            base_times_versus_time_analisado['delta'] = [2 if x>=2 else x for x in base_times_versus_time_analisado['delta']]
            base_times_versus_time_analisado['delta'] = [-2 if x<=-2 else x for x in base_times_versus_time_analisado['delta']]
        else:
            base_times_versus_time_analisado['delta'] = [(x-y) for x,y in zip(base_times_versus_time_analisado['pontuacao_corr'],base_times_versus_time_analisado['media_until'])]
        #base_times_versus_time_analisado['delta'] = (base_times_versus_time_analisado['pontuacao_corr'] - base_times_versus_time_analisado['media_until'])/(base_times_versus_time_analisado['media_until'])

        base_times_versus_time_analisado['local_op'] = ['CASA' if x=='FORA' else 'FORA' for x in base_times_versus_time_analisado['local']]

        media_delta_geral = base_times_versus_time_analisado.delta.mean()

        delta_agg = base_times_versus_time_analisado.groupby(['local_op'])['delta'].mean().reset_index()
        tabela_agg_delta = delta_agg.append(pd.DataFrame(data=[['GERAL',media_delta_geral]], columns = delta_agg.columns))
        tabela_agg_delta['posicao_nome'] = pos_
        tabela_agg_delta_agg_pos = tabela_agg_delta_agg_pos.append(tabela_agg_delta)

        #print (base_times_versus_time_analisado)


    tabela_formatted_for_bokeh = tabela_agg_delta_agg_pos.pivot('posicao_nome','local_op','delta').reset_index()


    dict_for_plot = dict({
        'ata': 1,
        'mei': 2,
        'lat': 3,
        'zag': 4,
        'gol': 5,
        'tec': 6,

    })

    tabela_formatted_for_bokeh = tabela_formatted_for_bokeh.iloc[tabela_formatted_for_bokeh['posicao_nome'].map(dict_for_plot).sort_values().index]

    dict_for_plot = dict({
        'ata': 'Ataque',
        'mei': 'Meio Campo',
        'lat': 'Lateral',
        'zag': 'Zagueiro',
        'gol': 'Goleiro',
        'tec': 'Tecnico',

    })

    tabela_formatted_for_bokeh['posicao_nome'] = tabela_formatted_for_bokeh['posicao_nome'].map(dict_for_plot)




    fruits = tabela_formatted_for_bokeh['posicao_nome'].tolist()
    years = ['CASA', 'FORA', 'GERAL']

    data = {'pos' : tabela_formatted_for_bokeh['posicao_nome'],
            'CASA'   : tabela_formatted_for_bokeh['CASA'],
            'FORA'   : tabela_formatted_for_bokeh['FORA'],
            'GERAL'   : tabela_formatted_for_bokeh['GERAL']}

    # this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
    x = [(fruit, year) for fruit in fruits for year in years ]
    counts = sum(zip(data['CASA'], data['FORA'], data['GERAL']), ()) # like an hstack


    color_1 = '#FF6D00'
    color_2 = '#240046'
    color_3 = '#9D4EDD'

    list_colors = []
    k = 1
    for v_ in x:

        if(k>3):
            k=1

        if(k==1):
            list_colors.append(color_1)
        
        if(k==2):
            list_colors.append(color_2)

        if(k==3):
            list_colors.append(color_3)

        k = k+1

    source = ColumnDataSource(data=dict(x=x, counts=counts, cr_ = list_colors))

    p = figure(x_range=FactorRange(*x), height=550, width=800,
            toolbar_location=None, tools="")
    from bokeh.palettes import Spectral6

    p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",

        # use the palette to colormap based on the the x[1:2] values
        #factor_cmap('x', palette=Spectral6, factors=years, start=1, end=2
        fill_color='cr_')

    list_x_vals = []
    list_x_offs = []
    mult_ = 4.4
    for z in range(1,len(tabela_formatted_for_bokeh)+1):
        list_x_vals.append(mult_*(z-1)+1-0.75)
        list_x_vals.append(mult_*(z-1)+1-0)
        list_x_vals.append(mult_*(z-1)+1+0.75)


    label = []
    for value_ in counts:
        if(flag_relative):
            label.append('{0:.0f}%'.format(value_*100))
        else:
            label.append('{0:.1f}'.format(value_))

    list_x_offs = []
    k = 1
    cycle_count = 1
    for w_, v_ in zip(counts, label):

        if(cycle_count>3):
            cycle_count = 1

        if(k>len(label)-3):
            list_x_offs.append(0)
        else:
            if(w_<0 and len(v_)>3):
                list_x_offs.append(18*(cycle_count-2))
            elif(w_<0 and len(v_)<=3):
                list_x_offs.append(18*(cycle_count-2))
            elif(w_>0 and len(v_)>3):
                list_x_offs.append(18*(cycle_count-2))
            elif(w_>0 and len(v_)<=3):
                list_x_offs.append(18*(cycle_count-2))

                
        
        cycle_count = cycle_count + 1
        
        k = k+1

    list_x_offs[-1] = 5

    y_offset_list=[20 if x<0 else -8 for x in counts]

    source_ = ColumnDataSource(dict(
                        text = label,
                        x_  = list_x_vals,
                        y_  =  counts,
                        x_off = list_x_offs,
                        y_off = y_offset_list,
                    ))

    p.text(x='x_', y='y_', text='text', source=source_, x_offset='x_off', y_offset='y_off', text_font_size='18px', text_font_style='bold')



    foto_url = df_clubes[df_clubes['id']==op_].escudo_url.tolist()[0]
    if(flag_relative):
        p.y_range = Range1d(min(counts)-0.25, max(counts)+1)
        
        source_ = ColumnDataSource(dict(
            url = [foto_url],
            x_  = [list_x_vals[-3]],
            y_  = [max(counts)+0.5]
        ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=90, h = 90, anchor="center")
    else:
        p.y_range = Range1d(min(counts)-4, max(counts)+4)
        source_ = ColumnDataSource(dict(
        url = [foto_url],
        x_  = [list_x_vals[-3]],
        y_  = [max(counts)+2]
    ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=90, h = 90, anchor="center")


    p.add_glyph(source_, image3)

    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.ygrid.grid_line_color = None

    if(flag_insta):
        p.xgrid.grid_line_color = '#b67be6'
        p.below[0].group_text_color = '#461e5c'
        p.xaxis.major_label_text_color = '#461e5c'
        p.background_fill_color = '#ffcf70'
        p.background_fill_alpha  = 0.5
        p.border_fill_color = '#ffcf70'
    else:
        p.xgrid.grid_line_color = '#ECEDDA'
        p.below[0].group_text_color = 'black'
        p.xaxis.major_label_text_color = 'black'
        p.background_fill_color = 'white'
        p.background_fill_alpha  = 0.5
        p.border_fill_color = 'white'

    p.xgrid.grid_line_alpha = 0.5
    p.xgrid.grid_line_dash = [6, 4]

    p.yaxis.minor_tick_out = 0
    p.yaxis.major_tick_out = 0
    p.xaxis.axis_line_color= None
    p.yaxis.axis_line_color= None
    p.outline_line_color = None

    p.yaxis.major_label_text_font_size = '0pt'
    p.xaxis.major_label_text_font_size = '10pt'
    p.below[0].group_text_font_size = '16px'

    p.yaxis.minor_tick_line_width = 0
    p.yaxis.major_tick_line_width = 0
    p.yaxis.minor_tick_out = 0

    return p


#%%

def get_table_resumo(team_id, df_clubes, df_data_scouts___, base_global_partidas, flag_insta=True):
    
    tabela_top_jogadores_versus = df_data_scouts___[(df_data_scouts___['opposing_team_id']==team_id) & (df_data_scouts___['posicao_nome'].isin(['ata','mei','zag','lat']))
    ].groupby(['rodada_id','posicao_nome'])['pontuacao'].max().reset_index().sort_values(by='rodada_id', ascending=False)

    tabela_top_jogadores_versus['opposing_team_id'] = team_id

    tabela_dados = pd.merge(tabela_top_jogadores_versus,
    df_data_scouts___[['opposing_team_id','rodada_id','pontuacao','posicao_nome','atleta_id','apelido','foto','clube_name','clube_id', 'local']],
    how ='left',
    on=['rodada_id','pontuacao','posicao_nome','opposing_team_id'])

    list_rods_ = tabela_dados['rodada_id'].unique().tolist()[:6]

    tabela_dados_ = tabela_dados[tabela_dados['rodada_id'].isin(list_rods_)]

    p = figure(x_range=Range1d(0,150), y_range=Range1d(-5,140),  height=700, width=900,
                toolbar_location=None, tools="")

    sequence_of_height = [110, 90, 70, 50, 30, 10]

    dict_positons = dict({

        'ata':{
            'fixed_x': 30,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        'mei':{
            'fixed_x': 60,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        'lat':{
            'fixed_x': 90,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        'zag':{
            'fixed_x': 120,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        }

    })

    base_placar_ = pd.merge(pd.merge(base_global_partidas,
    df_clubes[['id','abreviacao']],
    how='left',
    left_on='clube_casa_id',
        right_on='id').drop('id', axis=1)[['clube_casa_id',
        'clube_visitante_id',
        'placar_oficial_mandante',
        'placar_oficial_visitante',
        'abreviacao','rodada_id']].rename(columns={
            'abreviacao': 'abreviacao_casa'
        }),
    df_clubes[['id','abreviacao']],
    how='left',
    left_on='clube_visitante_id',
        right_on='id').drop('id', axis=1).rename(columns={
            'abreviacao': 'abreviacao_fora'
        })


    list_of_pos = ['ata','mei','zag','lat']
    color_ = ['lightgreen', "#cab2d6",'lightgreen', "#cab2d6",'lightgreen', "#cab2d6"]
    k = 0

    source = ColumnDataSource(dict(x=[75], y=[130], w=[150], h=[25]))
    if(flag_insta):
        glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 1, fill_color='#004aad', line_color=None)
    else:
        glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 1, fill_color='white', line_color=None)

    p.add_glyph(source, glyph)

    for rodada_, color_ in zip(tabela_dados_.rodada_id.unique().tolist(), color_):

        height_ = sequence_of_height[k]

        jogos_da_rodada = base_placar_[base_placar_['rodada_id']==rodada_]
        jogos_da_rodada_ = jogos_da_rodada[(jogos_da_rodada['clube_visitante_id']==team_id) | (jogos_da_rodada['clube_casa_id']==team_id)]
        str_ = str(jogos_da_rodada_.abreviacao_casa.tolist()[0]) + " " + str(jogos_da_rodada_.placar_oficial_mandante.tolist()[0]) + " vs "
        str_ = str_ + str(jogos_da_rodada_.placar_oficial_visitante.tolist()[0]) + " " + str(jogos_da_rodada_.abreviacao_fora.tolist()[0])

        tc = jogos_da_rodada_.clube_casa_id.tolist()[0]
        tv = jogos_da_rodada_.clube_visitante_id.tolist()[0]

        pc = jogos_da_rodada_.placar_oficial_mandante.tolist()[0]
        pv = jogos_da_rodada_.placar_oficial_visitante.tolist()[0]

        color_back = ""
        if((tc==team_id and pc>pv) or (tv==team_id and pc<pv)):
            color_back = '#58db84'
        elif((tc==team_id and pc<pv) or (tv==team_id and pc>pv)):
            color_back = '#fa3232'
        else:
            color_back = '#edb400'

        source = ColumnDataSource(dict(x=[75], y=[height_], w=[150], h=[25]))
        glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 0.6, fill_color=color_back, line_color=None)
        p.add_glyph(source, glyph)

        local_ = tabela_dados_[tabela_dados_['rodada_id']==rodada_].local.tolist()[0]
        local_ = 'CASA' if local_=='FORA' else 'FORA'

        source_ = ColumnDataSource(dict(
                            text = [local_],
                            x_  = [9],
                            y_  =  [height_-3]
                        ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')


        source_ = ColumnDataSource(dict(
                            text = ['Rodada {0}'.format(rodada_)],
                            x_  = [7],
                            y_  =  [height_]
                        ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')

        

        source_ = ColumnDataSource(dict(
                            text = [str_],
                            x_  = [4.5],
                            y_  =  [height_-6]
                        ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')

        for pos_player in list_of_pos:

            example_player = tabela_dados_[(tabela_dados_['rodada_id']==rodada_) &
                                            (tabela_dados_['posicao_nome']==pos_player)].atleta_id.tolist()[0]

            player_clube = df_data_scouts___[df_data_scouts___['atleta_id']==example_player].clube_id.tolist()[0]
            player_photo = df_data_scouts___[df_data_scouts___['atleta_id']==example_player].foto.tolist()[0]
            player_name = df_data_scouts___[df_data_scouts___['atleta_id']==example_player].apelido.tolist()[0]

            dados_na_rodada = df_data_scouts___[(df_data_scouts___['atleta_id']==example_player) & (df_data_scouts___['rodada_id']==rodada_)]
            pontuacao_decisivas = (dados_na_rodada['SG']*5 + dados_na_rodada['A']*5 + dados_na_rodada['G']*8).tolist()[0]
            pontuacao_basica = (dados_na_rodada['pontuacao'] - pontuacao_decisivas).tolist()[0]

            source_ = ColumnDataSource(dict(
                        url = [player_photo],
                        x_  = [dict_positons[pos_player]['fixed_x']],
                        y_  = [height_]
                    ))

            image3 = ImageURL(url='url',
            x='x_',
            y='y_',
            w_units='screen', h_units='screen', w=60, h = 60, anchor="center")

            p.add_glyph(source_, image3)


            source_ = ColumnDataSource(dict(
                                    text = ['PB: {0:.1f}'.format(pontuacao_basica)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pb']]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                                    text = ['PD: {0:.1f}'.format(pontuacao_decisivas)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pd']]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')


            source_ = ColumnDataSource(dict(
                                    text = ['{0:.1f}'.format(pontuacao_decisivas + pontuacao_basica)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']+1.5],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pt']]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                                    text = ['{0}'.format(player_name)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']-10],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pt']+5]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')
        
        
        k = k+1

    escudo_url = df_clubes[df_clubes['id']==team_id].escudo_url.tolist()[0]
    source_ = ColumnDataSource(dict(
                url = [escudo_url],
                x_  = [12.5],
                y_  = [127.5]
            ))

    image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
    p.add_glyph(source_, image3)

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None

    p.yaxis.minor_tick_line_width = 0
    p.yaxis.major_tick_line_width = 0
    p.yaxis.minor_tick_out = 0

    p.xaxis.minor_tick_line_width = 0
    p.xaxis.major_tick_line_width = 0
    p.xaxis.minor_tick_out = 0

    p.yaxis.major_label_text_font_size = '0pt'

    p.xaxis.major_label_text_font_size = '0pt'

    p.outline_line_color = None
    p.axis.axis_line_color=None

    return p

#%%

def get_table_resumo_atuante(team_id, df_clubes, df_data_scouts___, base_global_partidas, flag_insta=True):

    tabela_top_jogadores_versus = df_data_scouts___[(df_data_scouts___['clube_id']==team_id) & (df_data_scouts___['posicao_nome'].isin(['ata','mei','zag','lat']))
        ].groupby(['rodada_id','posicao_nome'])['pontuacao'].max().reset_index().sort_values(by='rodada_id', ascending=False)

    tabela_top_jogadores_versus['clube_id'] = team_id

    tabela_dados = pd.merge(tabela_top_jogadores_versus,
    df_data_scouts___[['opposing_team_id','rodada_id','pontuacao','posicao_nome','atleta_id','apelido','foto','clube_name','clube_id', 'local']],
    how ='left',
    on=['rodada_id','pontuacao','posicao_nome','clube_id'])

    list_rods_ = tabela_dados['rodada_id'].unique().tolist()[:6]

    tabela_dados_ = tabela_dados[tabela_dados['rodada_id'].isin(list_rods_)]

    p = figure(x_range=Range1d(0,150), y_range=Range1d(-5,140),  height=700, width=900,
                toolbar_location=None, tools="")

    sequence_of_height = [110, 90, 70, 50, 30, 10]

    dict_positons = dict({

        'ata':{
            'fixed_x': 30,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        'mei':{
            'fixed_x': 60,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        'lat':{
            'fixed_x': 90,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        'zag':{
            'fixed_x': 120,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        }

    })

    base_placar_ = pd.merge(pd.merge(base_global_partidas,
    df_clubes[['id','abreviacao']],
    how='left',
    left_on='clube_casa_id',
        right_on='id').drop('id', axis=1)[['clube_casa_id',
        'clube_visitante_id',
        'placar_oficial_mandante',
        'placar_oficial_visitante',
        'abreviacao','rodada_id']].rename(columns={
            'abreviacao': 'abreviacao_casa'
        }),
    df_clubes[['id','abreviacao']],
    how='left',
    left_on='clube_visitante_id',
        right_on='id').drop('id', axis=1).rename(columns={
            'abreviacao': 'abreviacao_fora'
        })


    list_of_pos = ['ata','mei','zag','lat']
    color_ = ['lightgreen', "#cab2d6",'lightgreen', "#cab2d6",'lightgreen', "#cab2d6"]
    k = 0

    source = ColumnDataSource(dict(x=[75], y=[130], w=[150], h=[25]))
    if(flag_insta):
        glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 1, fill_color='#004aad', line_color=None)
    else:
        glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 1, fill_color='white', line_color=None)

    p.add_glyph(source, glyph)

    for rodada_, color_ in zip(tabela_dados_.rodada_id.unique().tolist(), color_):

        height_ = sequence_of_height[k]

        jogos_da_rodada = base_placar_[base_placar_['rodada_id']==rodada_]
        jogos_da_rodada_ = jogos_da_rodada[(jogos_da_rodada['clube_visitante_id']==team_id) | (jogos_da_rodada['clube_casa_id']==team_id)]
        str_ = str(jogos_da_rodada_.abreviacao_casa.tolist()[0]) + " " + str(jogos_da_rodada_.placar_oficial_mandante.tolist()[0]) + " vs "
        str_ = str_ + str(jogos_da_rodada_.placar_oficial_visitante.tolist()[0]) + " " + str(jogos_da_rodada_.abreviacao_fora.tolist()[0])

        tc = jogos_da_rodada_.clube_casa_id.tolist()[0]
        tv = jogos_da_rodada_.clube_visitante_id.tolist()[0]

        pc = jogos_da_rodada_.placar_oficial_mandante.tolist()[0]
        pv = jogos_da_rodada_.placar_oficial_visitante.tolist()[0]

        color_back = ""
        if((tc==team_id and pc>pv) or (tv==team_id and pc<pv)):
            color_back = '#58db84'
        elif((tc==team_id and pc<pv) or (tv==team_id and pc>pv)):
            color_back = '#fa3232'
        else:
            color_back = '#edb400'

        source = ColumnDataSource(dict(x=[75], y=[height_], w=[150], h=[25]))
        glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 0.6, fill_color=color_back, line_color=None)
        p.add_glyph(source, glyph)

        local_ = tabela_dados_[tabela_dados_['rodada_id']==rodada_].local.tolist()[0]
        #local_ = 'CASA' if local_=='FORA' else 'FORA'

        source_ = ColumnDataSource(dict(
                            text = [local_],
                            x_  = [9],
                            y_  =  [height_-3]
                        ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')


        source_ = ColumnDataSource(dict(
                            text = ['Rodada {0}'.format(rodada_)],
                            x_  = [7],
                            y_  =  [height_]
                        ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')

        

        source_ = ColumnDataSource(dict(
                            text = [str_],
                            x_  = [4.5],
                            y_  =  [height_-6]
                        ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')

        for pos_player in list_of_pos:

            example_player = tabela_dados_[(tabela_dados_['rodada_id']==rodada_) &
                                            (tabela_dados_['posicao_nome']==pos_player)].atleta_id.tolist()[0]

            player_clube = df_data_scouts___[df_data_scouts___['atleta_id']==example_player].clube_id.tolist()[0]
            player_photo = df_data_scouts___[df_data_scouts___['atleta_id']==example_player].foto.tolist()[0]
            player_name = df_data_scouts___[df_data_scouts___['atleta_id']==example_player].apelido.tolist()[0]

            dados_na_rodada = df_data_scouts___[(df_data_scouts___['atleta_id']==example_player) & (df_data_scouts___['rodada_id']==rodada_)]
            pontuacao_decisivas = (dados_na_rodada['SG']*5 + dados_na_rodada['A']*5 + dados_na_rodada['G']*8).tolist()[0]
            pontuacao_basica = (dados_na_rodada['pontuacao'] - pontuacao_decisivas).tolist()[0]

            source_ = ColumnDataSource(dict(
                        url = [player_photo],
                        x_  = [dict_positons[pos_player]['fixed_x']],
                        y_  = [height_]
                    ))

            image3 = ImageURL(url='url',
            x='x_',
            y='y_',
            w_units='screen', h_units='screen', w=60, h = 60, anchor="center")

            p.add_glyph(source_, image3)


            source_ = ColumnDataSource(dict(
                                    text = ['PB: {0:.1f}'.format(pontuacao_basica)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pb']]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                                    text = ['PD: {0:.1f}'.format(pontuacao_decisivas)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pd']]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='12px', text_font_style='bold')


            source_ = ColumnDataSource(dict(
                                    text = ['{0:.1f}'.format(pontuacao_decisivas + pontuacao_basica)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']+1.5],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pt']]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                                    text = ['{0}'.format(player_name)],
                                    x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']-10],
                                    y_  =  [height_ + dict_positons[pos_player]['delta_y_pt']+5]
                                ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')
        
        
        k = k+1

    escudo_url = df_clubes[df_clubes['id']==team_id].escudo_url.tolist()[0]
    source_ = ColumnDataSource(dict(
                url = [escudo_url],
                x_  = [12.5],
                y_  = [127.5]
            ))

    image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
    p.add_glyph(source_, image3)

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None

    p.yaxis.minor_tick_line_width = 0
    p.yaxis.major_tick_line_width = 0
    p.yaxis.minor_tick_out = 0

    p.xaxis.minor_tick_line_width = 0
    p.xaxis.major_tick_line_width = 0
    p.xaxis.minor_tick_out = 0

    p.yaxis.major_label_text_font_size = '0pt'

    p.xaxis.major_label_text_font_size = '0pt'

    p.outline_line_color = None
    p.axis.axis_line_color=None

    return p

def get_plot_saldo_de_gols(df_data_scouts___, df_clubes, local = ['CASA','FORA'], flag_half = 'top', flag_insta = False):
    
    df_sg = df_data_scouts___[df_data_scouts___['local'].isin(local)].groupby(['clube_id', 'clube_name','rodada_id'])['SG'].sum().reset_index()
    df_sg['SG'] = ['SG_' if x > 0 else 'NSG_' for x in df_sg['SG'].tolist()]

    if not(flag_insta):
        flag_half = 'all'

    df_sg = df_sg.drop(['rodada_id'], axis=1).groupby(['clube_name','SG']).size(
    ).reset_index().pivot('clube_name','SG',0).reset_index().fillna(0)

    df_sg['tot_'] = df_sg['NSG_'] + df_sg['SG_']

    df_sg['label_pos_top'] = df_sg['NSG_'] + df_sg['SG_']/2

    df_sg['label_pos_bottom'] = df_sg['NSG_']/2

    df_sg['perc_'] = df_sg['SG_']/df_sg['tot_']

    df_sg = df_sg.sort_values(by='perc_', ascending=False)


    if(flag_half=='top'):
        df_sg = df_sg.head(10)
    elif(flag_half=='bottom'):
        df_sg = df_sg.tail(10)
    else:
        df_sg = df_sg

    df_sg['label_pos_perc'] = max(df_sg.tot_.tolist())*1.1

    df_sg['perc_label'] = ['{0:.0f}%'.format(x*100) for x in df_sg['perc_'].tolist()]

    df_sg['label_pos_image'] = max(df_sg.tot_.tolist())*1.38

    df_sg = pd.merge(df_sg,df_clubes[['nome','escudo_url']],how='left',left_on='clube_name', right_on='nome')

    source = ColumnDataSource(df_sg)
    f = figure(x_range=df_sg.clube_name.tolist(),
                    y_range=Range1d(0,max(df_sg.tot_.tolist())*1.7),
                    plot_height=500,
                    plot_width = 1300)

    s = f.vbar(x='clube_name', bottom=0, top='NSG_', width=0.5, source=source, color='#FF5C4D')
    p = f.vbar(x='clube_name', bottom='NSG_', top='tot_', width=0.5, source=source, color='#97E855')

    f.hex(x='clube_name', y='label_pos_bottom', size=40, source=source, color='#010100', fill_alpha=0.5)
    f.hex(x='clube_name', y='label_pos_top', size=40, source=source, color='#010100', fill_alpha=0.5)
    

    if not(flag_insta):
        f.text(x='clube_name', y='label_pos_perc', source=source, text='perc_label', x_offset=-15, y_offset=0, text_font_size='15pt')
        image2 = ImageURL(url="escudo_url", x="clube_name", y='label_pos_image', w_units='screen', h_units='screen', w=60, h = 60, anchor='center')
        p_label = f.text(x='clube_name',
                        y='label_pos_top',
                        source=source, text='SG_', x_offset=-8, y_offset = 10, text_color='white', text_font_size='15pt')

        s_label = f.text(x='clube_name', y='label_pos_bottom',
                        source=source, text='NSG_', x_offset=-8, y_offset = 10, text_color='white', text_font_size='13pt')
    else:
        f.text(x='clube_name', y='label_pos_perc', source=source, text='perc_label', x_offset=-20, y_offset=+15, text_font_size='24pt', text_font_style='bold')
        image2 = ImageURL(url="escudo_url", x="clube_name", y='label_pos_image', w_units='screen', h_units='screen', w=80, h = 80, anchor='center')

        s_label = f.text(x='clube_name', y='label_pos_bottom',
                        source=source, text='NSG_', x_offset=-12, y_offset = 13, text_color='white', text_font_size='20pt')


        p_label = f.text(x='clube_name',
                        y='label_pos_top',
                        source=source, text='SG_', x_offset=-12, y_offset = 13, text_color='white', text_font_size='20pt')


    f.add_glyph(source, image2)

    f.xaxis.major_label_orientation = math.pi/4

    f.ygrid.grid_line_color = None
    f.xgrid.grid_line_color = None
    f.yaxis.minor_tick_line_width = 0
    f.yaxis.major_tick_line_width = 0
    f.yaxis.minor_tick_out = 1
    f.ygrid.grid_line_color = None
    f.outline_line_color = None
    f.axis.axis_line_color=None
    f.yaxis.major_label_text_font_size = '0pt'
    f.xaxis.major_label_text_font_size = '0pt'

    legend = Legend(items=[(fruit, [r]) for (fruit, r) in zip(['Sem SG','Com SG'], [s,p])], location=(-150, 425))

    local_titulo = 'GERAL' if len(local)==2 else local[0].upper()

    if not(flag_insta):
        f.add_layout(legend, 'right')

        f.add_layout(Title(text='% de SG jogando em {0}'.format(local_titulo.upper()), align="center"), "above")

    return f

#%%

def get_round_plot_cartolafc(df_data_scouts___, df_clubes,  team_nomes, rodada_atual_num, min_num_jogos=5):
    
    pl = df_data_scouts___.groupby('atleta_id')['rodada_id'].count().reset_index()
    pl = pl[pl['rodada_id']>=min_num_jogos]['atleta_id'].tolist()
    df_bayern = df_data_scouts___[(df_data_scouts___['clube_name']==team_nomes) &
                                    (df_data_scouts___['rodada_id']>=(rodada_atual_num-10)) &
                                    (df_data_scouts___['atleta_id'].isin(pl))].groupby(['atleta_id','apelido']).agg(
                                        posicao_nome=('posicao_nome','first'),
                                                                                        player_photo = ('foto','first'),
                                                                                        pg_ = ('pontuacao', 'mean'),
                                                                                            pc_ = ('pontuacao','std'),
                                                                                                nj_ = ('rodada_id','count')).reset_index(
                                                                                                ).sort_values(by='nj_', ascending=False).head(20).sort_values(by='pg_',
                                                                                                ascending=False).reset_index().head(10)


    df_baseline = df_data_scouts___[
    (df_data_scouts___['rodada_id']>=(rodada_atual_num-10)) &
    (df_data_scouts___['atleta_id'].isin(pl))].groupby(['atleta_id','apelido','posicao_nome']).agg(
    player_photo = ('foto','first'),
    pg_base = ('pontuacao', 'mean'),
    pc_base = ('pontuacao','std'),
    nj_ = ('rodada_id','count')).reset_index(
    ).sort_values(by='pc_base', ascending=False).groupby(['posicao_nome']).agg(
    pg_base = ('pg_base', lambda x: x.quantile(0.80)),
    pc_base = ('pc_base', lambda x: x.quantile(0.80))
    ).reset_index()

    #df_baseline['pc_base'] = df_baseline['pc_base']*1.25

    df_bayern = pd.merge(df_bayern, df_baseline, how='left', on='posicao_nome')

    def get_back_color(m,d,mbase,dbase):

        if(m>=mbase and d>dbase):
            return '#F9E79F'
        elif(m>=mbase and d<=dbase):
            return '#73C6B6'
        elif(m<mbase and d>dbase):
            return '#F5B7B1'
        elif(m<mbase and d<=dbase):
            return '#A9CCE3'

    df_bayern['back_color'] = df_bayern.apply(lambda x: get_back_color(x['pg_'],
                                                                    x['pc_'],
                                                                    x['pg_base'],
                                                                    x['pc_base']), axis=1)

    depara_clube_logo = df_clubes[['escudo_url','id','nome']].drop_duplicates(keep='first')
    logo = df_clubes[df_clubes['nome']==team_nomes]['escudo_url']
    nome = df_clubes[df_clubes['nome']==team_nomes]['nome'].tolist()[0]


    drug_color = OrderedDict([
        ("pg_", "#0d3362"),
        ("pc_", "#c64737")
    ])

    gram_color = OrderedDict([
    ("negative", "#e69584"),
    ("positive", "#aeaeb8"),
    ])


    df = df_bayern.copy()

    width = 650
    height = 650
    inner_radius = 30
    outer_radius = inner_radius + 15

    minr = sqrt(log(0.1 * 1E4))
    maxr = sqrt(log(1000 * 1E4))
    a = (outer_radius - inner_radius) / (minr - maxr)
    b = inner_radius - a * maxr

    def rad(mic):
        return mic + inner_radius

    big_angle = 2.0 * np.pi / (len(df) + 1)
    small_angle = big_angle / 5

    p = figure(plot_width=width, plot_height=height, title="",
        x_axis_type=None, y_axis_type=None,
        x_range=(-60, 60), y_range=(-60, 60),
        min_border=0, outline_line_color="black",
        background_fill_color=None)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # annular wedges
    angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
    colors = [gram for gram in df.back_color]
    p.annular_wedge(
        0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors, fill_alpha=0.9
    )

    # small wedges
    p.annular_wedge(0, 0, inner_radius, rad(df.pg_*0.8),
                    -big_angle+angles+3*small_angle, -big_angle+angles+4*small_angle,
                    color=drug_color['pg_'])

    p.annular_wedge(0, 0, inner_radius,rad(df.pc_*0.8),
                    -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
                    color=drug_color['pc_'])


    # circular axes and lables
    labels = np.power(10.0, np.arange(1, 2))
    radii = []
    radii.append(inner_radius + 10)
    p.circle(0, 0, radius=inner_radius + 10, fill_color=None, line_color="white")
    p.text(0, inner_radius + 10, ['10'],
            text_font_size="11px", text_align="center", text_baseline="middle")

    source_ = ColumnDataSource(dict(
        url = [r"C:\Users\thiag\Pictures\Imagem1.png"],
        x_  = [0],
        y_  = [inner_radius + 10]
    ))

    image3 = ImageURL(url='url', x='x_', y='y_', w=14, h = 14, anchor="center")
    p.add_glyph(source_, image3)

    source_l = ColumnDataSource(dict(
        url = [logo],
        x_  = [0],
        y_  = [0]
    ))

    image4 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=90, h = 90, anchor="center")
    p.add_glyph(source_l, image4)

    p.text(0, inner_radius + 20, ['Cartologia'],
            text_font_size="20px", text_align="center", text_baseline="middle")



    # radial axes
    p.annular_wedge(0, 0, inner_radius-5, outer_radius+10,
                    -big_angle+angles, -big_angle+angles, color="black")

    # bacteria labels
    lang_ = 0.88
    xr = radii[0]*np.cos(np.array(-big_angle*lang_ + angles))*1.15
    yr = radii[0]*np.sin(np.array(-big_angle*lang_+ angles))*1.15
    label_angle=np.array(-big_angle*lang_+angles)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, df.apelido, angle=label_angle,
            text_font_size="16px", text_align="center", text_baseline="middle", text_font_style='italic')


    lang_ = 0.4
    xr = radii[0]*np.cos(np.array(-big_angle*lang_ + angles))*1.3
    yr = radii[0]*np.sin(np.array(-big_angle*lang_ + angles))*1.3
    label_angle=np.array(-big_angle*lang_+angles)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side

    xr[0] = radii[0]*np.cos(np.array(-big_angle*lang_ + angles[0]))*1.45
    yr[0] = radii[0]*np.sin(np.array(-big_angle*lang_ + angles[0]))*1.45

    source = ColumnDataSource(dict(
        url = df.player_photo.tolist(),
        x_  = xr,
        y_  = yr,
        angle_ = label_angle
    ))

    image2 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=85, h = 85, anchor="center")
    p.add_glyph(source, image2)

    # bacteria labels
    xr = rad(df.pg_*0.8)*np.cos(np.array(-big_angle+angles+3.5*small_angle))*1.10
    yr = rad(df.pg_*0.8)*np.sin(np.array(-big_angle+angles+3.5*small_angle))*1.10
    label_angle=np.array(-big_angle+angles+3.5*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, ['{0:.1f}'.format(x) for x in df.pg_.tolist()], angle=label_angle,
            text_font_size="16px", text_align="center", text_baseline="middle", text_font_style='bold')


    # bacteria labels
    xr = rad(df.pc_*0.8)*np.cos(np.array(-big_angle+angles+1.5*small_angle))*1.10
    yr = rad(df.pc_*0.8)*np.sin(np.array(-big_angle+angles+1.5*small_angle))*1.10
    label_angle=np.array(-big_angle+angles+1.5*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, ['{0:.1f}'.format(x) for x in df.pc_.tolist()], angle=label_angle,
            text_font_size="16px", text_align="center", text_baseline="middle", text_font_style='bold')


    # OK, these hand drawn legends are pretty clunky, will be improved in future release
    p.circle([-20, -20], [-370, -390], color=list(gram_color.values()), radius=5)
    p.text([-40, -30], [-370, -390], text=["Gram-" + gr for gr in gram_color.keys()],
            text_font_size="9px", text_align="left", text_baseline="middle")

    p.rect([0, 0], [14, -14], width=35, height=5,
            color=list(drug_color.values()))
    p.text([0, 0], [14, -14], text=['Média', 'Desvio Padrão'], text_color='white',
            text_font_size="18px", text_align="center", text_baseline="middle", text_font_style='bold')

    p.outline_line_color = None
    p.background_fill_color = 'white'
    p.border_fill_color = 'white'
    p.outline_line_color = 'white'

    return p

#%%

def get_pass_network_on_pitch_plot(rodada_atual_num, depara_times_footstats, base_global_partidas, df_geodata_total, df_data_scouts___,
                base_ids, df_atletas, base_interacao_scouts, team_to_see, scout_name__, percentile_filter = 0.75, flag_filter_local = '', flag_insta_params = True, flag_hist = False):
    
    if(flag_insta_params):
        angle_media = 1.57
        x_off_media = -35
        y_off_media = 39
    else:
        angle_media = 0
        x_off_media = 0
        y_off_media = -32

    team_id___ = depara_times_footstats[depara_times_footstats['team_name']==team_to_see].team_id.tolist()[0]

    fix_list_home = base_global_partidas[base_global_partidas['home_team_id']==team_id___].fix_id.tolist()
    fix_list_away = base_global_partidas[base_global_partidas['away_team_id']==team_id___].fix_id.tolist()

    df_geodata_total_ = pd.merge(df_geodata_total,base_global_partidas[['fix_id','rod_num']],how='left', on='fix_id')
    df_geodata_total_ = df_geodata_total_[df_geodata_total_['rod_num']>=(rodada_atual_num-14)]

    condition_home = df_geodata_total_['fix_id'].isin(fix_list_home)
    condition_away = df_geodata_total_['fix_id'].isin(fix_list_away)
    condition_all = df_geodata_total_['fix_id'].isin(fix_list_away) | df_geodata_total_['fix_id'].isin(fix_list_home)
    condition_team = df_geodata_total_['team_id']==team_id___

    if(flag_filter_local=='casa'):
        condition_final = condition_team & condition_home
    elif(flag_filter_local=='fora'):
        condition_final = condition_team & condition_away
    else:
        condition_final = condition_team & condition_all

    local_campo = pd.merge(df_geodata_total_[condition_final].groupby(['player_id','player_name']).agg(
        n_jogos_ = ('fix_id', pd.Series.nunique),
        inter_x_ = ('inter_x', 'median'),
        inter_y_ = ('inter_y', 'median'),
        inter_tot = ('player_id', 'count')
    ).reset_index(), base_ids[['idPlayer','atleta_id']], how='left', left_on='player_id', right_on='idPlayer').sort_values(by=['n_jogos_','inter_tot'], ascending=False)

    dp_scouts_pos = df_data_scouts___[['atleta_id','posicao_nome','foto']].drop_duplicates(subset=['atleta_id','posicao_nome'], keep='first')
    dp_scouts_pos['atleta_id'] = dp_scouts_pos['atleta_id'].astype(int)

    local_campo_ = pd.merge(local_campo, dp_scouts_pos, how='left', on='atleta_id')

    gol = local_campo_[local_campo_['posicao_nome']=='gol'].head(1)

    jogadores_provaveis_duvida = df_atletas
    jogadores_provaveis_duvida['atleta_id'] = jogadores_provaveis_duvida['atleta_id'].astype(int)

    local_campo_ = pd.merge(local_campo_, jogadores_provaveis_duvida[['atleta_id','status_id']], how='left', on='atleta_id')
    local_campo_ = local_campo_[local_campo_['status_id'].isin([2,7,6,5])]

    dict_sorting_status = dict({
        2: '2 - Duvida',
        7: '1 - Provavel',
        6: '3 - Nulo',
        5: '5 - Lesão'
    })

    local_campo_['status_id'] = local_campo_['status_id'].map(dict_sorting_status)

    if(flag_hist):
        local_campo_ = local_campo_.sort_values(by=['n_jogos_','inter_tot'], ascending=False)
    else:
        local_campo_ = local_campo_.sort_values(by='status_id', ascending=True)

    #print(local_campo_)

    zags = local_campo_[local_campo_['posicao_nome']=='zag'].head(2)
    lats = local_campo_[local_campo_['posicao_nome']=='lat'].head(2)
    mei = local_campo_[local_campo_['posicao_nome']=='mei'].head(4)
    ata = local_campo_[local_campo_['posicao_nome']=='ata'].head(2)

    line_up = gol.append(zags).append(lats).append(mei).append(ata)

    local_campo_ = line_up.sort_values(by=['inter_x_','posicao_nome'])

    p = aux_func.draw_pitch(width = 700, height = 500,
                        measure = 'metres',
                        fill_color = '#D1F9AF', fill_alpha = 0.4,
                        line_color = '#2E5A0A', line_alpha = 0.6,
                        hspan = [-52.5, 52.5], vspan = [-34, 34],
                        arcs = True)

    source = ColumnDataSource(local_campo_)

    glyph = Circle(x="inter_x_", y="inter_y_", size=1, line_color="#3288bd", fill_color="white", line_width=3)
    p.add_glyph(source, glyph)

    p.text(source=source, x='inter_x_', y='inter_y_', text='player_name', y_offset=y_off_media, x_offset=x_off_media, angle=angle_media, text_font_style='bold')


    if(flag_filter_local=='casa'):
        network_passes_ = base_interacao_scouts[(base_interacao_scouts['scout_name']==scout_name__) &
    (base_interacao_scouts['team_name']==team_to_see) &
    (base_interacao_scouts['fix_id'].isin(fix_list_home))].groupby(['idPlayer', 'nomePlayer', 'team_id',
        'idPlayerDestino', 'nome','team_name'])['qtd'].sum().reset_index()

    elif(flag_filter_local=='fora'):
        network_passes_ = base_interacao_scouts[(base_interacao_scouts['scout_name']==scout_name__) &
        (base_interacao_scouts['team_name']==team_to_see) &
        (base_interacao_scouts['fix_id'].isin(fix_list_away))].groupby(['idPlayer', 'nomePlayer', 'team_id',
            'idPlayerDestino', 'nome','team_name'])['qtd'].sum().reset_index()
    else:
        network_passes_ = base_interacao_scouts[(base_interacao_scouts['scout_name']==scout_name__) &
        (base_interacao_scouts['team_name']==team_to_see)].groupby(['idPlayer', 'nomePlayer', 'team_id',
            'idPlayerDestino', 'nome','team_name'])['qtd'].sum().reset_index()

    list_top_participantes = local_campo_.player_id.tolist()

    network_passes_ = network_passes_[(network_passes_['idPlayer'].isin(list_top_participantes)) &
    (network_passes_['idPlayerDestino'].isin(list_top_participantes))].sort_values(by='qtd', ascending=False)


    max_qtd_ = int(network_passes_.qtd.max())
    min_qtd_ = network_passes_.qtd.min()


    network_passes_['line_wid'] = 6*network_passes_['qtd']/network_passes_['qtd'].max()

    red = Color("red")
    colors = list(red.range_to(Color("green"),max_qtd_))

    vec_c = []
    k = 1
    for color_ in colors:
        color_rgb = color_.get_hex()
        vec_c.append([k, color_rgb])
        k = k+1

    base_cores_ = pd.DataFrame(columns = ['qtd','color_'], data = vec_c)

    network_passes_ = pd.merge(network_passes_, base_cores_, how='left', on='qtd')

    mediana_ = network_passes_['qtd'].quantile(percentile_filter)

    network_passes_ = network_passes_[network_passes_['qtd']>=mediana_]

    for current_player, player_picture in zip(local_campo_.player_id.tolist(), local_campo_.foto.tolist()):

        network_passes_player = network_passes_[network_passes_['idPlayer']==current_player]

        base_network = pd.merge(network_passes_player,
            local_campo_[['player_id','inter_x_', 'inter_y_']],
            how='left',
            left_on='idPlayer',
            right_on='player_id').rename(columns = {'inter_x_':'inter_x_origem', 'inter_y_':'inter_y_origem'}).drop('player_id', axis=1)

        base_network = pd.merge(base_network,
            local_campo_[['player_id','inter_x_', 'inter_y_']],
            how='left',
            left_on='idPlayerDestino',
            right_on='player_id').rename(columns = {'inter_x_':'inter_x_destino', 'inter_y_':'inter_y_destino'})


        source = ColumnDataSource(base_network)

        arrow = Arrow(x_start='inter_x_origem',
        x_end='inter_x_destino',
        y_start='inter_y_origem',
        y_end='inter_y_destino',
            line_width='line_wid',
            line_color='color_',
            source=source,
                start=NormalHead(fill_color="black", size=0),
                end=NormalHead(fill_color="black", size=0), level='underlay')

        p.add_layout(arrow)

        x__ = local_campo_[local_campo_['player_id']==current_player].inter_x_.tolist()[0]
        y__ = local_campo_[local_campo_['player_id']==current_player].inter_y_.tolist()[0]

        source_ = ColumnDataSource(dict(
                url = [player_picture],
                x_  = [x__],
                y_  = [y__]
            ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=60, h = 60, anchor="center", angle=angle_media)
        p.add_glyph(source_, image3)

        depara_team_logos = base_global_partidas[['home_team_id', 'home_team_logo']].drop_duplicates()
        escudo_url = depara_team_logos[depara_team_logos['home_team_id']==team_id___].home_team_logo.tolist()[0]

    source_ = ColumnDataSource(dict(
            url = [escudo_url],
            x_  = [-45],
            y_  = [27.5]
        ))

    image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center", angle=angle_media)
    p.add_glyph(source_, image3)

    if(flag_filter_local!='casa' and flag_filter_local!='fora'):
        flag_filter_local='geral'


    source_ = ColumnDataSource(dict(
            url = ["(" + flag_filter_local.upper() + ")"],
            x_  = [-36.5],
            y_  = [23]
        ))

    glyph = Text(x="x_", y="y_", text="url",
    text_font_style='bold',
                                        text_color= "darkgreen",
                                        text_font_size= "16px", angle=angle_media)

    p.add_glyph(source_, glyph)

    return p