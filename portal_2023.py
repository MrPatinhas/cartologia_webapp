#%%
from re import X
from threading import local
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
import pandas as pd
import urllib, json
import requests
import aux_func
from tqdm import tqdm
from bokeh.transform import factor_cmap
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
import io
from PIL import Image
import os
import requests

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

from colour import Color
import colorsys
from os import listdir
from os.path import isfile, join

from bokeh.models import (Arrow, ColumnDataSource, CustomJS, Label,
                        NormalHead, SingleIntervalTicker, TapTool)

from bokeh.io.export import get_screenshot_as_png
from streamlit_lottie import st_lottie

#%%

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def get_basic_lineup():
    
    df_confrontos_summerized = pd.merge(pd.merge(df_confrontos_rodada_atual, df_clubes[['id', 'nome', 'escudo_url']], how='left', left_on='clube_casa_id', right_on='id').rename(columns={
        'id':'id_casa',
        'nome': 'nome_time_casa',
        'escudo_url': 'escudo_casa'
    }), df_clubes[['id', 'nome', 'escudo_url']], how='left', left_on='clube_visitante_id', right_on='id').rename(columns={
        'id':'id_fora',
        'nome': 'nome_time_fora',
        'escudo_url': 'escudo_fora'
    })

    sequence_of_height = [125, 95, 65, 35, 5]

    dict_positons = dict({

        1:{
            'fixed_x': 35,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        2:{
            'fixed_x': 55,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        3:{
            'fixed_x': 95,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        },
        4:{
            'fixed_x': 115,
            'delta_label_pontucao_x': 7.25,
            'delta_rect_x': 5,
            'delta_y_pt': 3,
            'delta_y_pb': -2,
            'delta_y_pd': -7
        }

    })

    list_of_pos = ['ata','mei','zag','lat']
    color_ = ['lightgreen', "#cab2d6",'lightgreen', "#cab2d6",'lightgreen', "#cab2d6"]
    k = 0



    p = figure(x_range=Range1d(10,130), y_range=Range1d(-15,140),  height=700, width=750,
                    toolbar_location=None, tools="")

    line_reset = 0

    for k_index in range(0, len(df_confrontos_summerized)):

        height_ = sequence_of_height[int(k_index/2)]

        nome_home = df_confrontos_summerized.nome_time_casa.tolist()[k_index]
        
        escudo_home = df_confrontos_summerized.escudo_casa.tolist()[k_index]
        nome_away = df_confrontos_summerized.nome_time_fora.tolist()[k_index]
        
        escudo_away = df_confrontos_summerized.escudo_fora.tolist()[k_index]

        color__ = 'gray' #green' if diff_norm_list_val >= 20/100 else 'yellow' if diff_norm_list_val>=10/100 else 'gray'

        #flag_arrow_home = r'C:\Users\thiag\Documents\Python Scripts\Sheets Fun\up_arrow.png' if df_confrontos_total__[df_confrontos_total__['team_name']==nome_home].diff_norm.tolist()[0] > 0 else r'C:\Users\thiag\Documents\Python Scripts\Sheets Fun\down_arrow.png'
        #flag_arrow_away = r'C:\Users\thiag\Documents\Python Scripts\Sheets Fun\up_arrow.png' if flag_arrow_home == r'C:\Users\thiag\Documents\Python Scripts\Sheets Fun\down_arrow.png' else r'C:\Users\thiag\Documents\Python Scripts\Sheets Fun\down_arrow.png'
        
        nome_away = nome_away.replace(" ", " \n ")
        nome_home = nome_home.replace(" ", " \n ")

        if(k_index%2==0):

            source = ColumnDataSource(dict(x=[dict_positons[1]['fixed_x']+10], y=[height_-2.5], w=[40], h=[28]))
            glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 0.2, fill_color=color__, line_color=None)

            p.add_glyph(source, glyph)

            source_ = ColumnDataSource(dict(
                                text = [nome_home],
                                x_  = [dict_positons[1]['fixed_x']-7],
                                y_  =  [height_-16.5]
                            ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                        url = [escudo_home],
                        x_  = [dict_positons[1]['fixed_x']],
                        y_  = [height_]
                    ))

            image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
            p.add_glyph(source_, image3)

            #source_ = ColumnDataSource(dict(
            #            url = [flag_arrow_home],
            #            x_  = [dict_positons[1]['fixed_x']+7.5],
            #            y_  = [height_+6.5]
            #        ))#

            #image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=25, h = 25, anchor="center")
            #p.add_glyph(source_, image3)

            source_ = ColumnDataSource(dict(
                                text = [nome_away],
                                x_  = [dict_positons[2]['fixed_x']-7],
                                y_  =  [height_-16.5]
                            ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                                text = ['Vs'],
                                x_  = [(dict_positons[2]['fixed_x']+dict_positons[1]['fixed_x'])/2-1],
                                y_  =  [height_-4]
                            ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                        url = [escudo_away],
                        x_  = [dict_positons[2]['fixed_x']],
                        y_  = [height_]
                    ))

            image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
            p.add_glyph(source_, image3)

            #source_ = ColumnDataSource(dict(
            #            url = [flag_arrow_away],
            #            x_  = [dict_positons[2]['fixed_x']+7.5],
            #            y_  = [height_+6.5]
            #        ))

            #image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=25, h = 25, anchor="center")
            #p.add_glyph(source_, image3)

        else:

            source = ColumnDataSource(dict(x=[dict_positons[3]['fixed_x']+10], y=[height_-2.5], w=[40], h=[28]))
            glyph = Rect(x="x", y="y", width="w", height="h", angle=0, fill_alpha = 0.2, fill_color=color__, line_color=None)

            p.add_glyph(source, glyph)
            
            source_ = ColumnDataSource(dict(
                                text = [nome_home],
                                x_  = [dict_positons[3]['fixed_x']-7],
                                y_  =  [height_-16.5]
                            ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                        url = [escudo_home],
                        x_  = [dict_positons[3]['fixed_x']],
                        y_  = [height_]
                    ))

            image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
            p.add_glyph(source_, image3)

            #source_ = ColumnDataSource(dict(
            #            url = [flag_arrow_home],
            #            x_  = [dict_positons[3]['fixed_x']+7.5],
            #            y_  = [height_+6.5]
            #        ))

            #image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=25, h = 25, anchor="center")
            #p.add_glyph(source_, image3)

            source_ = ColumnDataSource(dict(
                                text = [nome_away],
                                x_  = [dict_positons[4]['fixed_x']-7],
                                y_  =  [height_-16.5]
                            ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                                text = ['Vs'],
                                x_  = [(dict_positons[3]['fixed_x']+dict_positons[4]['fixed_x'])/2-1],
                                y_  =  [height_-4]
                            ))

            p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')

            source_ = ColumnDataSource(dict(
                        url = [escudo_away],
                        x_  = [dict_positons[4]['fixed_x']],
                        y_  = [height_]
                    ))

            image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=80, h = 80, anchor="center")
            p.add_glyph(source_, image3)

            #source_ = ColumnDataSource(dict(
            #            url = [flag_arrow_away],
            #            x_  = [dict_positons[4]['fixed_x']+7.5],
            #            y_  = [height_+6.5]
            #        ))

            #image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=25, h = 25, anchor="center")
            #p.add_glyph(source_, image3)
        
        #if(k_index>=3):
        #    break

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

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_most_similar_match(ft_name, ft_clube, iteration_unification, df_data_scouts___):
    
    base_cartola = df_data_scouts___[['atleta_id','apelido', 'clube_id']].drop_duplicates(subset='atleta_id', keep='last')
    
    base_cartola_restante = base_cartola[(~base_cartola['atleta_id'].isin(
        iteration_unification.atleta_id.unique().tolist())) & 
                                        (base_cartola['clube_id']==ft_clube)]
    vec_ = []
    for k in range(0, len(base_cartola_restante)):
        nome_  = base_cartola_restante.apelido.tolist()[k]
        player_id  = base_cartola_restante.atleta_id.tolist()[k]
        team_name  = base_cartola_restante.clube_id.tolist()[k]
        ratio_ = similar(ft_name, nome_)

        vec_.append([nome_, player_id, team_name, ratio_])

    dataframe_semelhanca = pd.DataFrame(data = vec_, columns = ['nome_', 'atleta_id', 'team_name', 'ratio_'])
    
    values_ = dataframe_semelhanca.sort_values(by='ratio_', ascending=False).head(1)
    if(len(values_)>0):
        if(values_.ratio_.tolist()[0]>0.4):
            return values_.atleta_id.tolist()[0]
        else:
            return "-"
    else:
        return "-"

def get_pass_network_on_pitch_plot(team_to_see, scout_name__,
 percentile_filter = 0.75, flag_filter_local = '',
  flag_insta_params = True, flag_hist = False, num_matches_back=1):
    
    if(flag_insta_params):
        angle_media = 1.57
        x_off_media = -35
        y_off_media = 39
    else:
        angle_media = 0
        x_off_media = 0
        y_off_media = -32

    team_id___ = depara_times_footstats[depara_times_footstats['team_name']==team_to_see].team_id_footstats.tolist()[0]

    fix_list_home = base_global_partidas[base_global_partidas['home_team_id']==team_id___].fix_id.tolist()
    fix_list_away = base_global_partidas[base_global_partidas['away_team_id']==team_id___].fix_id.tolist()

    df_geodata_total_ = pd.merge(df_geodata_total,base_global_partidas[['fix_id','rod_num']],how='left', on='fix_id')
    df_geodata_total_ = df_geodata_total_[df_geodata_total_['rod_num']>=(rodada_atual_num-num_matches_back)]

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
        inter_x_ = ('inter_x',  'mean'),
        inter_y_ = ('inter_y',  'mean'),
        inter_tot = ('player_id', 'count')
    ).reset_index(), base_data_foodstat_[['idPlayer','atleta_id']], how='left', left_on='player_id', right_on='idPlayer').sort_values(by=['n_jogos_','inter_tot'], ascending=False)

    dp_scouts_pos = df_data_scouts___[['atleta_id','posicao_nome','foto']].drop_duplicates(subset=['atleta_id','posicao_nome'], keep='first')
    dp_scouts_pos['atleta_id'] = dp_scouts_pos['atleta_id'].astype(int)

    local_campo = local_campo[~local_campo['atleta_id'].isnull()]


    local_campo['atleta_id'] = local_campo['atleta_id'].astype(int)


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
    mei = local_campo_[local_campo_['posicao_nome'].isin(['mei','ata'])].head(6)

    line_up = gol.append(zags).append(lats).append(mei)

    local_campo_ = line_up.sort_values(by=['inter_x_','posicao_nome'])

    print(local_campo_)

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
        depara_team_logos_aux = base_global_partidas[['away_team_id', 'away_team_logo']].drop_duplicates().rename(columns={'away_team_id':'home_team_id', 'away_team_logo': 'home_team_logo'})
        depara_team_logos = depara_team_logos.append(depara_team_logos_aux).drop_duplicates()
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

def get_depara_from_data_sources(df_data_scouts___, base_interacao_scouts, depara_times_footstats, base_scouts):

    base_footStats = pd.merge(base_interacao_scouts[['idPlayer','nomePlayer', 'team_id']].drop_duplicates(),
            base_scouts[['team_id','team_name']].drop_duplicates(),
            how='left',
            on='team_id')

    base_footStats = pd.merge(base_footStats.rename(columns = {'team_id': 'team_id_footstats'}), depara_times_footstats.drop('team_name', axis=1),
    how='left', on='team_id_footstats')

    iteration_unification = pd.merge(base_footStats, df_data_scouts___[['atleta_id','apelido', 'clube_id']].drop_duplicates(subset='apelido', keep='last'), how='left', left_on=['nomePlayer','clube_id'],
                                    right_on=['apelido','clube_id'])

    unmatch = iteration_unification[(pd.isnull(iteration_unification['apelido'])) &
                        (iteration_unification['team_name'].isin(depara_times_footstats.team_name.unique()))]

    unmatch['atleta_id'] = unmatch.apply(lambda x: get_most_similar_match(x['nomePlayer'], x['clube_id'], iteration_unification, df_data_scouts___), axis=1)

    unmatch_matched = unmatch[unmatch['atleta_id']!="-"]

    base_data_foodstat_ = iteration_unification[~((pd.isnull(iteration_unification['apelido'])) &
                        (iteration_unification['team_name'].isin(depara_times_footstats.team_name.unique())))].append(unmatch_matched)

    base_data_foodstat_ = base_data_foodstat_.drop('apelido', axis=1)

    return base_data_foodstat_

def get_json_data_heatmapa(idAway_, idHome_, fix_id_):
    

    url = 'https://footstatsapiapp.azurewebsites.net//partidas/heatmapByTeam'

    data = {'idAway': idAway_, 'idChampionship': id_championship_footstats, 'idHome': idHome_, 'idMatch': fix_id_}

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

def get_geopostion_dataframe():

    df_geodata_total = pd.DataFrame(columns=['team_id', 'player_id', 'player_name', 'inter_x', 'inter_y', 'inter_tempo', 'fix_id'])

    for home_team_id, away_team_id, fix_id_ in tqdm(zip(base_global_partidas['home_team_id'].tolist(),
                                                base_global_partidas['away_team_id'].tolist(),
                                                base_global_partidas['fix_id'].tolist())):


        df_geo_position = get_json_data_heatmapa(away_team_id, home_team_id, fix_id_)
        df_geodata_total = df_geodata_total.append(df_geo_position)
    
    return df_geodata_total

def get_most_similar_match_fbref(ft_name, ft_clube, iteration_unification, df_data_scouts___):
    
    base_cartola = df_data_scouts___[~df_data_scouts___['posicao_nome'].isin(['tec', 'gol'])][['atleta_id','apelido', 'clube_name']].drop_duplicates(subset='atleta_id', keep='last')
    
    base_cartola_restante = base_cartola[(~base_cartola['atleta_id'].isin(
        iteration_unification.atleta_id.unique().tolist())) & 
                                        (base_cartola['clube_name']==ft_clube)]
    vec_ = []
    for k in range(0, len(base_cartola_restante)):
        nome_  = base_cartola_restante.apelido.tolist()[k]
        player_id  = base_cartola_restante.atleta_id.tolist()[k]
        team_name  = base_cartola_restante.clube_name.tolist()[k]
        ratio_ = similar(ft_name, nome_)

        vec_.append([nome_, player_id, team_name, ratio_])

    dataframe_semelhanca = pd.DataFrame(data = vec_, columns = ['nome_', 'atleta_id', 'clube_name', 'ratio_'])
    
    values_ = dataframe_semelhanca.sort_values(by='ratio_', ascending=False).head(1)
    if(len(values_)>0):
        if(values_.ratio_.tolist()[0]>0.4):
            return values_.atleta_id.tolist()[0]
        else:
            return "-"
    else:
        return "-"
    
def get_depara_table_fbref(global_line_data, df_data_scouts___):
        
    base_fbref = global_line_data[['player_name', 'clube_name']].drop_duplicates()

    iteration_unification = pd.merge(base_fbref, df_data_scouts___[~df_data_scouts___['posicao_nome'].isin(['tec', 'gol'])][['atleta_id','apelido', 'clube_name']].drop_duplicates(subset='apelido', keep='last'), how='left', left_on=['player_name','clube_name'],
                                    right_on=['apelido','clube_name'])

    unmatch = iteration_unification[(pd.isnull(iteration_unification['apelido']))]

    unmatch['atleta_id'] = unmatch.apply(lambda x: get_most_similar_match_fbref(x['player_name'], x['clube_name'], iteration_unification, df_data_scouts___), axis=1)

    unmatch_matched = unmatch[unmatch['atleta_id']!="-"]

    base_data_fbref_ = iteration_unification[~(pd.isnull(iteration_unification['apelido']))].append(unmatch_matched)

    base_data_fbref__ = base_data_fbref_.drop('apelido', axis=1)

    return base_data_fbref__

def get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'progressive_ball_carries',
title_suggested = 'Conduções Progressivas',
agg_operation = 'sum',
flag_level_aggregation = 'apelido',
top_x = 10, flag_filter_position = '', flag_filter_team = '', flag_position=''):

    if(metric_to_plot in df_data_scouts___.columns):
        df_data_table = df_data_scouts___
    else:
        df_data_table = pd.merge(df_data_scouts___, global_line_data_[['atleta_id', 'rod_num', metric_to_plot]], how='left', 
                                left_on=['atleta_id', 'rodada_id'], right_on=['atleta_id', 'rod_num'])

    
    filter_position = [True]*len(df_data_table) if flag_filter_position== "" else df_data_table['posicao_nome']==flag_filter_position

    filter_team = [True]*len(df_data_table) if flag_filter_team== "" else df_data_table['clube_name']==flag_filter_team

    df_data_table = df_data_table[filter_position & filter_team]


    if(flag_level_aggregation=='clube_name'):

        db_agg_generic = df_data_table.groupby([flag_level_aggregation]).agg(

            metric_to_plot_agg = (metric_to_plot, agg_operation),
            qtd_ocorrencias = ('rodada_id', pd.Series.nunique)

        ).reset_index()

        db_agg_generic['metric_by_game'] = db_agg_generic['metric_to_plot_agg']/db_agg_generic['qtd_ocorrencias']

    else:

        db_agg_generic = df_data_table.groupby([flag_level_aggregation]).agg(
            foto = ('foto', 'last'),
            clube_name = ('clube_name', 'last'),
            metric_to_plot_agg = (metric_to_plot, agg_operation),
            qtd_ocorrencias = ('rodada_id', pd.Series.nunique)

        ).reset_index()

        db_agg_generic['metric_by_game'] = db_agg_generic['metric_to_plot_agg']/db_agg_generic['qtd_ocorrencias']

    df_table = pd.merge(db_agg_generic, df_clubes[['nome', 'escudo_url']].rename(columns = {'nome': 'clube_name'}), 
                                        how='left', on='clube_name').sort_values(by='metric_to_plot_agg', ascending=False).head(top_x)


    df_table['player_photo_pos'] = [df_table['metric_to_plot_agg'].max()*1.2 for x in df_table['metric_to_plot_agg'].tolist()]

    df_table['clube_photo_pos'] = [df_table['metric_to_plot_agg'].max()*1.8 for x in df_table['metric_to_plot_agg'].tolist()]

    df_table['label_pos_data'] = [x*0.9 for x in df_table['metric_to_plot_agg'].tolist()]

    if(df_table.metric_to_plot_agg.apply(float.is_integer).all()):
        df_table['label_data'] = ['{0:.0f}'.format(x) for x in df_table['metric_to_plot_agg'].tolist()]
    else:
        df_table['label_data'] = ['{0:.1f}'.format(x) for x in df_table['metric_to_plot_agg'].tolist()]

    df_table['label_pos_data_2'] = [x for x in df_table['metric_by_game'].tolist()] 
    df_table['label_data_2'] = ['{0:.1f}'.format(x) for x in df_table['metric_by_game'].tolist()]

    if(flag_level_aggregation=='clube_name'):
        df_table['x_axis_label'] = [x.replace(" ", "\n",1).replace("-","\n",1) for x in df_table['clube_name']]
    else:
        df_table['x_axis_label'] = [x.replace(" ", "\n",1) for x in df_table['apelido']]

    if(flag_level_aggregation!='clube_name'):
        f = figure(x_range=df_table.x_axis_label.tolist(),
                y_range=Range1d(0,max(df_table['metric_to_plot_agg'].tolist())*2),
                height=600,
                width = 800)
    else:
        f = figure(x_range=df_table.x_axis_label.tolist(),
                y_range=Range1d(0,max(df_table['metric_to_plot_agg'].tolist())*1.5),
                height=600,
                width = 800)
    source = ColumnDataSource(df_table)

    b1 = f.vbar(x='x_axis_label', bottom=0, top='metric_to_plot_agg', width=0.75, source=source, color='#FFCD58', line_color='#FFCD58', line_alpha=0.1, line_width=5, fill_alpha=0.8)
    f.hex(x='x_axis_label', y='label_pos_data', size=40, source=source, color='#010100', fill_alpha=0.5)
    f.text(x='x_axis_label', y='label_pos_data',
            source=source, text='label_data',
            x_offset=-9, y_offset=+10,
            text_color='white', text_font_size='14pt')

    f.yaxis.axis_label = 'Qtd de {0}'.format(title_suggested) if agg_operation=='sum' else 'Média de {0}'.format(title_suggested)
    f.yaxis.axis_label_text_font_size = "16pt"

    if(flag_level_aggregation!='clube_name'):
        image2 = ImageURL(url="foto", x="x_axis_label", y='player_photo_pos',
                            w_units ='screen',
                            h_units = 'screen',
                            w = 85,
                            h = 85,
                            anchor = "center")
        f.add_glyph(source, image2)

        image2 = ImageURL(url="escudo_url", x="x_axis_label", y='clube_photo_pos', w_units ='screen',
                            h_units = 'screen',
                            w = 75,
                            h = 75,
                            anchor = "center")

        f.add_glyph(source, image2)
    else:
        image2 = ImageURL(url="escudo_url", x="x_axis_label", y='player_photo_pos',
                            w_units ='screen',
                            h_units = 'screen',
                            w = 75,
                            h = 75,
                            anchor = "center")
        f.add_glyph(source, image2)

    # Setting the second y axis range name and range
    l_2 = df_table['metric_by_game'].tolist()
    f.extra_y_ranges = {"foo": Range1d(start=min(l_2)*0, end=max(l_2)*3)}

    f.xaxis.axis_label_text_font_size = "70pt"
    f.axis.axis_label_text_font_style = 'bold'
    f.xaxis.major_label_text_font_size = "10pt"
    f.xaxis.major_label_text_font_style = "bold"
    f.xaxis.minor_tick_line_width = 0
    f.xaxis.major_tick_line_width = 0
    f.xaxis.minor_tick_out = 1

    f.yaxis.minor_tick_line_width = 0
    f.yaxis.major_tick_line_width = 0
    f.yaxis.minor_tick_out = 1

    f.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels


    f.xaxis.major_label_standoff = 15

    f.ygrid.grid_line_color = None
    f.outline_line_color = None
    f.axis.axis_line_color=None
    f.xaxis.major_label_text_font_size = "14px"
    f.xgrid.grid_line_dash  = 'dashed'
    f.toolbar.logo = None
    f.toolbar_location = None

    return f

def get_generic_horizontal_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'progressive_ball_carries',
title_suggested = 'Conduções Progressivas',
agg_operation = 'sum',
flag_level_aggregation = 'apelido',
top_x = 10, flag_filter_position = '', flag_normalize_by_game=False, flag_filter_team=""):

    if(metric_to_plot in df_data_scouts___.columns):
        df_data_table = df_data_scouts___
    else:
        df_data_table = pd.merge(df_data_scouts___, global_line_data_[['atleta_id', 'rod_num', metric_to_plot]], how='left', 
                                left_on=['atleta_id', 'rodada_id'], right_on=['atleta_id', 'rod_num'])

    
    filter_position = [True]*len(df_data_table) if flag_filter_position== "" else df_data_table['posicao_nome']==flag_filter_position
    filter_team = [True]*len(df_data_table) if flag_filter_team== "" else df_data_table['clube_name']==flag_filter_team

    df_data_table = df_data_table[filter_position & filter_team]

    if(flag_level_aggregation=='clube_name'):

        db_agg_generic = df_data_table.groupby([flag_level_aggregation]).agg(

            metric_to_plot_agg = (metric_to_plot, agg_operation),
            qtd_ocorrencias = ('rodada_id', pd.Series.nunique)

        ).reset_index()

        db_agg_generic['metric_by_game'] = db_agg_generic['metric_to_plot_agg']/db_agg_generic['qtd_ocorrencias']

    else:

        db_agg_generic = df_data_table.groupby([flag_level_aggregation]).agg(
            foto = ('foto', 'last'),
            clube_name = ('clube_name', 'last'),
            metric_to_plot_agg = (metric_to_plot, agg_operation),
            qtd_ocorrencias = ('rodada_id', pd.Series.nunique)

        ).reset_index()

        db_agg_generic['metric_by_game'] = db_agg_generic['metric_to_plot_agg']/db_agg_generic['qtd_ocorrencias']

    if(flag_normalize_by_game):
        db_agg_generic['metric_to_plot_agg'] = db_agg_generic['metric_by_game']

    df_table = pd.merge(db_agg_generic, df_clubes[['nome', 'escudo_url']].rename(columns = {'nome': 'clube_name'}), 
                                        how='left', on='clube_name').sort_values(by='metric_to_plot_agg', ascending=False).head(top_x).sort_values(by='metric_to_plot_agg', ascending=True)


    df_table['player_photo_pos'] = [df_table['metric_to_plot_agg'].max()*1.2 for x in df_table['metric_to_plot_agg'].tolist()]

    df_table['clube_photo_pos'] = [df_table['metric_to_plot_agg'].max()*1.8 for x in df_table['metric_to_plot_agg'].tolist()]

    df_table['label_pos_data'] = [x*0.9 for x in df_table['metric_to_plot_agg'].tolist()]
    if(df_table.metric_to_plot_agg.apply(float.is_integer).all()):
        df_table['label_data'] = ['{0:.0f}'.format(x) for x in df_table['metric_to_plot_agg'].tolist()]
    else:
        df_table['label_data'] = ['{0:.1f}'.format(x) for x in df_table['metric_to_plot_agg'].tolist()]

    df_table['label_pos_data_2'] = [x for x in df_table['metric_by_game'].tolist()] 
    df_table['label_data_2'] = ['{0:.1f}'.format(x) for x in df_table['metric_by_game'].tolist()]

    if(flag_level_aggregation=='clube_name'):
        df_table['x_axis_label'] = [x.replace(" ", "\n",1).replace("-","\n",1) for x in df_table['clube_name']]
    else:
        df_table['x_axis_label'] = [x.replace(" ", "\n",1) for x in df_table['apelido']]

    if(flag_level_aggregation!='clube_name'):
        f = figure(y_range=df_table.x_axis_label.tolist(),
                x_range=Range1d(0,max(df_table['metric_to_plot_agg'].tolist())*2),
                height=700,
                width = 600)
    else:
        f = figure(y_range=df_table.x_axis_label.tolist(),
                x_range=Range1d(0,max(df_table['metric_to_plot_agg'].tolist())*1.5),
                height=700,
                width = 600)
    source = ColumnDataSource(df_table)

    b1 = f.hbar(y='x_axis_label', left=0, right='metric_to_plot_agg', height=0.65, source=source, color='#FFCD58', line_color='#FFCD58', line_alpha=0.1, line_width=5, fill_alpha=0.8)
    f.hex(y='x_axis_label', x='label_pos_data', size=40, source=source, color='#010100', fill_alpha=0.5)

    if(df_table.metric_to_plot_agg.apply(float.is_integer).all()):
        f.text(y='x_axis_label', x='label_pos_data',
                source=source, text='label_data',
                x_offset=-9, y_offset=+10,
                text_color='white', text_font_size='14pt')
    else:
        f.text(y='x_axis_label', x='label_pos_data',
                source=source, text='label_data',
                x_offset=-12, y_offset=+10,
                text_color='white', text_font_size='14pt')
    

    f.xaxis.axis_label = 'Qtd de {0}'.format(title_suggested) if agg_operation=='sum' else 'Média de {0}'.format(title_suggested)
    f.xaxis.axis_label_text_font_size = "12pt"

    if(flag_level_aggregation!='clube_name'):
        image2 = ImageURL(url="foto", y="x_axis_label", x='player_photo_pos',
                            w_units ='screen',
                            h_units = 'screen',
                            w = 65,
                            h = 65,
                            anchor = "center")
        f.add_glyph(source, image2)

        image2 = ImageURL(url="escudo_url", y="x_axis_label", x='clube_photo_pos', w_units ='screen',
                            h_units = 'screen',
                            w = 55,
                            h = 55,
                            anchor = "center")

        f.add_glyph(source, image2)
    else:
        image2 = ImageURL(url="escudo_url", y="x_axis_label", x='player_photo_pos',
                            w_units ='screen',
                            h_units = 'screen',
                            w = 60,
                            h = 60,
                            anchor = "center")
        f.add_glyph(source, image2)

    f.axis.axis_label_text_font_style = 'bold'
    f.xaxis.major_label_text_font_size = "0pt"
    f.xaxis.major_label_text_font_style = "bold"
    f.xaxis.minor_tick_line_width = 0
    f.xaxis.major_tick_line_width = 0
    f.xaxis.minor_tick_out = 1

    f.yaxis.minor_tick_line_width = 0
    f.yaxis.major_tick_line_width = 0
    f.yaxis.minor_tick_out = 1

    f.yaxis.major_label_text_font_size = '12pt'  # preferred method for removing tick labels
    f.yaxis.major_label_text_font_style = 'bold'

    f.xaxis.major_label_standoff = 15

    #f.ygrid.grid_line_color = None
    f.outline_line_color = None
    f.axis.axis_line_color=None
    f.xaxis.major_label_text_font_size = "0px"
    f.xgrid.grid_line_color  = None
    f.ygrid.grid_line_dash  = 'dashed'
    f.toolbar.logo = None
    f.toolbar_location = None

    return f

def get_distribution_points_plot(df_data_scouts___, player_id):

    df_player = df_data_scouts___[df_data_scouts___['atleta_id']==player_id]
    foto_url = df_data_scouts___[df_data_scouts___['atleta_id']==player_id].foto.unique().tolist()[0]

    df_agg = df_player.groupby('class_pontuacao').size().reset_index().sort_values(by='class_pontuacao',ascending=True)

    dict_labels = dict({
        'Mitou': ' E - Acima de 10',
        'Bom': 'D - Entre 5 e 10',
        'Médio': 'C - Entre 2 e 5',
        'Ruim': 'B - Entre 0 e 2',
        'Péssimo': 'A - Abaixo de 0'
    })

    df_agg['label'] = df_agg['class_pontuacao'].map(dict_labels)

    df_agg['perc'] = df_agg[0]*100/df_agg[0].sum()
    df_agg['perc'] = df_agg['perc'].astype(int)
    df_agg['label_text'] = ['{0}%'.format(x) for x in df_agg['perc']]
    df_agg['text_pos'] = df_agg['perc'] + 3

    df_agg = df_agg.rename(columns = {0:'qtd'})
    df_agg['label'] = df_agg['label'].astype(str)
    df_agg['class_pontuacao'] = df_agg['class_pontuacao'].astype(str)
    df_agg = df_agg.sort_values(by='label')

    p = figure(x_range=df_agg.label.tolist(),
                    y_range=Range1d(0,min(max(df_agg['perc'].tolist())*1.5, 150)),
                    height = 600,
                    width = 800)
    source = ColumnDataSource(df_agg)

    b1 = p.vbar(x='label', top='perc', width=0.75, source=source, color='#FFCD58', line_color='#FFCD58', line_alpha=0.1, line_width=5, fill_alpha=0.8)

    labels = LabelSet(x='label', y='text_pos', text='label_text',
            x_offset=-10, y_offset=0, source=source, text_font_size="15pt", text_color='#110F1A')

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



            source = ColumnDataSource(dict(x=[k+0.3+0.2],
                                            y=[(place)*incr+0.8], w=[0.6], h=[2*max(df_agg['perc'])/22]))

            #print(2*max(df_agg['perc'])/25)


            glyph = Rect(x="x", y="y", width="w", height="h", fill_color=color_, fill_alpha=0.6, line_alpha=0)
            p.add_glyph(source, glyph)

            source = ColumnDataSource(dict(x=[k+0.2],
            y=[(place)*incr], text=['Rod {0:.0f}'.format(rod)]))

            glyph = Text(x="x", y="y", text="text",
                                        text_color= "white",
                                        text_font_size= "18px")

            p.add_glyph(source, glyph)

            place = place + 1

    source = ColumnDataSource(dict(x=[3.75],
                                            y=[max(df_agg['text_pos'])*1.25+0.8], w=[1.8], h=[2*max(df_agg['perc'])/22]))


    glyph = Rect(x="x", y="y", width="w", height="h", fill_color='#725ac1', fill_alpha=0.6, line_alpha=0)
    p.add_glyph(source, glyph)

    source = ColumnDataSource(dict(x=[3],
                                            y=[max(df_agg['text_pos'])*1.25], text=['Rodada em Casa']))

    glyph = Text(x="x", y="y", text="text",
                                text_color= "white",
                                text_font_size= "18px")

    p.add_glyph(source, glyph)

    source = ColumnDataSource(dict(x=[3.75],
                                            y=[max(df_agg['text_pos'])*1.15+0.8], w=[1.8], h=[2*max(df_agg['perc'])/22]))


    glyph = Rect(x="x", y="y", width="w", height="h", fill_color='#242038', fill_alpha=0.6, line_alpha=0)
    p.add_glyph(source, glyph)

    source = ColumnDataSource(dict(x=[3],
                                            y=[max(df_agg['text_pos'])*1.15], text=['Rodada Fora']))

    glyph = Text(x="x", y="y", text="text",
                                text_color= "white",
                                text_font_size= "18px")

    p.add_glyph(source, glyph)


    source_ = ColumnDataSource(dict(
            url = [foto_url],
            x_  = [0.75],
            y_  = [max(df_agg['text_pos'])*1.2]
        ))

    image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=110, h = 110, anchor="center")
    p.add_glyph(source_, image3)

    return p

def get_table_resumo_atuante(team_id, flag_insta=True):

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

    base_placar_ = pd.merge(pd.merge(base_global_partidas_og,
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

            try:

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
                w_units='screen', h_units='screen', w=65, h = 65, anchor="center")

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
                                        y_  =  [height_ + dict_positons[pos_player]['delta_y_pt']-1.5]
                                    ))

                p.text(x='x_', y='y_', text='text', source=source_, text_font_size='20px', text_font_style='bold')

                source_ = ColumnDataSource(dict(
                                        text = ['{0}'.format(player_name)],
                                        x_  = [dict_positons[pos_player]['fixed_x'] + dict_positons[pos_player]['delta_label_pontucao_x']-10],
                                        y_  =  [height_ + dict_positons[pos_player]['delta_y_pt']+5]
                                    ))

                p.text(x='x_', y='y_', text='text', source=source_, text_font_size='15px', text_font_style='bold')
            except:
                pass
        
        
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
                    height=500,
                    width = 1300)

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

    if not(flag_insta):
        f.add_layout(legend, 'right')

        f.add_layout(Title(text='% de SG jogando em {0}'.format(local.upper()), align="center"), "above")

    return f

def get_round_plot_cartolafc(df_data_scouts___, df_clubes,  team_nomes, min_num_jogos=5):
    
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

    df_bayern = pd.merge(df_bayern, df_baseline, how='left', on='posicao_nome').fillna(0)

    def get_back_color(m,d,mbase,dbase):

        if(m>=mbase and d>dbase):
            return '#F9E79F'
        elif(m>=mbase and d<=dbase):
            return '#73C6B6'
        elif(m<=mbase and d>dbase):
            return '#F5B7B1'
        elif(m<=mbase and d<=dbase):
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

    p = figure(width=width, height=height, title="",
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

@stream.cache(allow_output_mutation=True)
def get_full_datatables_required():

    aux_rod_num = pd.read_csv('aux_rod_num.csv')
    rodada_atual_num = aux_rod_num.rod_num.tolist()[0]

    df_confrontos_rodada_atual = pd.read_csv('{0}_rod{1}.csv'.format('df_confrontos_rodada_atual', rodada_atual_num-1))
    df_atletas  = pd.read_csv('{0}_rod{1}.csv'.format('df_atletas', rodada_atual_num-1))
    df_data_scouts___ = pd.read_csv('{0}_rod{1}.csv'.format('df_data_scouts___', rodada_atual_num-1))
    df_status = pd.read_csv('{0}_rod{1}.csv'.format('df_status', rodada_atual_num-1))
    df_posicoes = pd.read_csv('{0}_rod{1}.csv'.format('df_posicoes', rodada_atual_num-1))
    base_global_partidas_og = pd.read_csv('{0}_rod{1}.csv'.format('base_global_partidas_og', rodada_atual_num-1))
    df_clubes = pd.read_csv('{0}_rod{1}.csv'.format('df_clubes', rodada_atual_num-1))
    dataframe_player = pd.read_csv('{0}_rod{1}.csv'.format('dataframe_player', rodada_atual_num-1))

    global_line_data_ = pd.read_csv('fbref_data_rod{0}.csv'.format(rodada_atual_num-1))
    global_gk_data_ = pd.read_csv('fbref_data_gk_rod{0}.csv'.format(rodada_atual_num-1))

    id_championship_footstats = 850

    league_name_at_file = 'brasileiro_2023_'

    dict_time_name_conversion_original = dict({
            'Palmeiras': 'Palmeiras',
            'Cuiabá': 'Cuiabá',
            'Corinthians': 'Corinthians',
            'Cruzeiro': 'Cruzeiro',
            'Red Bull Bragantino': 'Bragantino',
            'Bahia': 'Bahia',
            'Flamengo':  'Flamengo',
            'Coritiba': 'Coritiba',
            'Atlético-MG': 'Atlético-MG',
            'Vasco': 'Vasco',
            'Botafogo': 'Botafogo',
            'São Paulo': 'São Paulo',
            'Grêmio': 'Grêmio',
            'Santos': 'Santos',
            'Athletico-PR': 'Athlético-PR',
            'Goiás': 'Goiás',
            'Fortaleza': 'Fortaleza',
            'Internacional': 'Internacional',
            'América-MG': 'América-MG',
    'Fluminense': 'Fluminense'
    })

    base_scouts = pd.read_csv('base_scouts_footstats_{0}.csv'.format(league_name_at_file))
    base_interacao_scouts = pd.read_csv('base_scouts_footstats_interacao_{0}.csv'.format(league_name_at_file))
    base_global_partidas = pd.read_csv('base_partidas_{0}.csv'.format(league_name_at_file))
    depara_times_footstats = base_scouts[['team_id','team_name']].drop_duplicates().rename(columns = {'team_id': 'team_id_footstats'})
    df_tt = df_clubes[['id', 'nome']].drop_duplicates()

    dict_time_name_conversion = {v: k for k, v in dict_time_name_conversion_original.items()}
    df_tt['nome'] = df_tt['nome'].map(dict_time_name_conversion)
    df_tt = df_tt.rename(columns={'id': 'clube_id', 'nome':'clube_name'})
    depara_times_footstats = pd.merge(depara_times_footstats, df_tt, how='left', left_on='team_name', right_on='clube_name')
    base_data_foodstat_ = get_depara_from_data_sources(df_data_scouts___, base_interacao_scouts, depara_times_footstats, base_scouts)
    df_geodata_total = pd.read_csv('pass_network_database_{0}.csv'.format(rodada_atual_num-1))

    tabela_times = pd.merge(base_data_foodstat_[['team_id_footstats','team_name', 'clube_id']].drop_duplicates(),
    df_data_scouts___[['clube_id','clube_name']].drop_duplicates(),
    how='left', on='clube_id').sort_values(by='clube_id')

    ata_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'ata', alpha_risk=0.5, beta_risk=0.25, min_num_jogos = int(rodada_atual_num/3))
    lat_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'lat', alpha_risk=0.5, beta_risk=0.25, min_num_jogos = int(rodada_atual_num/3))
    zag_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'zag', alpha_risk=0.5, beta_risk=0.25, min_num_jogos = int(rodada_atual_num/3))
    mei_ = aux_func.dataframe_best_players_in_round(rodada_atual_num, df_atletas, df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'mei', alpha_risk=0.5, beta_risk=0.25, min_num_jogos = int(rodada_atual_num/3))
    full_players_rank = ata_.append(mei_).append(zag_).append(lat_).sort_values(by='score_pond', ascending=False)

    dataframe_player['atleta_id'] = dataframe_player['atleta_id'].astype(int)
    df_data_scouts___['atleta_id'] = df_data_scouts___['atleta_id'].astype(int)

    return tabela_times, df_geodata_total, base_global_partidas, base_interacao_scouts, base_scouts, global_line_data_, df_confrontos_rodada_atual, df_atletas, df_data_scouts___, df_status, df_posicoes, base_global_partidas_og, rodada_atual_num, df_clubes, dataframe_player, depara_times_footstats, base_data_foodstat_, full_players_rank

#%%

def get_plot_points_by_scout_breakdown(df_data_scouts___, atl_id, flag_show_player=False):
    
    df_data_scouts___cp = df_data_scouts___.copy()
    df_data_scouts___cp['F'] = 0.8*df_data_scouts___cp['FF'] + 1.2*df_data_scouts___cp['FD'] + 3*df_data_scouts___cp['FT']
    df_data_scouts___cp['G'] = 8*df_data_scouts___cp['G']
    df_data_scouts___cp['A'] = 5*df_data_scouts___cp['A']
    df_data_scouts___cp['DS'] = 1.2*df_data_scouts___cp['DS']
    df_data_scouts___cp['FS'] = 0.5*df_data_scouts___cp['FS']
    df_data_scouts___cp['SG'] = 5*df_data_scouts___cp['SG']
    df_data_scouts___cp_plot = df_data_scouts___cp[['rodada_id', 'atleta_id', 'apelido', 'foto', 'pontuacao',
        'posicao_id', 'posicao_nome', 'clube_id', 'clube_name','DS', 'F', 'G', 'A', 'FS', 'SG']]

    df_table = df_data_scouts___cp_plot[(df_data_scouts___cp_plot['atleta_id']==atl_id) &
                                        (df_data_scouts___cp_plot['rodada_id']>=(rodada_atual_num-7))]
    df_table['y0'] = df_table['DS']
    df_table['y1'] = df_table['y0']+df_table['F']
    df_table['y2'] = df_table['y1']+df_table['G']
    df_table['y3'] = df_table['y2']+df_table['A']
    df_table['y4'] = df_table['y3']+df_table['FS']
    df_table['y5'] = df_table['y4']+df_table['SG']
    df_table['y6'] = df_table['y5'] + 1*max((max(df_table.pontuacao.tolist())/10),1)
    df_table['y7'] = df_table['y5'] + 0.75*max(1,((max(df_table.pontuacao.tolist())/10)))

    df_table['lDS'] = ['' if x ==0 else '{0:.2f}'.format(x) for x in df_table['DS']]
    df_table['lF'] = ['' if x ==0 else '{0:.2f}'.format(x) for x in df_table['F']]
    df_table['lG'] = ['' if x ==0 else '{0:.2f}'.format(x) for x in df_table['G']]
    df_table['lA'] = ['' if x ==0 else '{0:.2f}'.format(x) for x in df_table['A']]
    df_table['lFS'] = ['' if x ==0 else '{0:.2f}'.format(x) for x in df_table['FS']]
    df_table['lSG'] = ['' if x ==0 else '{0:.2f}'.format(x) for x in df_table['SG']]
    df_table['lpontuacao'] = ['' if x ==0 else '{0:.2f}'.format(x)[:4] for x in df_table['pontuacao']]

    source_imgs = ColumnDataSource(df_table)
    data = df_table.to_dict(orient='list')

    fruits = data['rodada_id']
    years = ['DS', 'F', 'G', 'A', 'FS', 'SG']

    p = figure(x_range=Range1d(rodada_atual_num-9,max(df_table.rodada_id.tolist())+0.5), y_range=Range1d(0,max(10, max(df_table.pontuacao.tolist()) + 8)), height=550, width=725,
            toolbar_location=None)

    p.vbar_stack(years, x='rodada_id', width=0.7, source=data, color = ['#89CFF0', 'red', 'lightgreen', 'yellow', 'pink', 'orange'],
                legend_label=years, fill_alpha=0.7)

    p.circle(x='rodada_id', y='y7', size=40, color='gray', line_color=None, fill_alpha=0.5, source=data)

    for n,s in enumerate(['DS', 'F', 'G', 'A', 'FS', 'SG', 'pontuacao']):

        labels = LabelSet(x='rodada_id', y='y{0}'.format(n), text='l{0}'.format(s), level='glyph', text_font_style = 'bold',
                x_offset=-13.5, y_offset=-20 + (max((max(df_table.pontuacao.tolist())/5),1)-1), source=source_imgs)

        p.add_layout(labels)
    if(flag_show_player):
        source_ = ColumnDataSource(dict(
                                url = [df_table.foto.tolist()[0]],
                                x_  = [rodada_atual_num-8.3],
                                y_  = [max(10, max(df_table.pontuacao.tolist())+8) - 2.5*max((max(df_table.pontuacao.tolist())/10),1)]
                            ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=100, h = 100, anchor="center")
        p.add_glyph(source_, image3)

        source_ = ColumnDataSource(dict(
                                        text = [df_table.apelido.tolist()[0]],
                                        x_  = [rodada_atual_num-8.3],
                                        y_  =  [max(10, max(df_table.pontuacao.tolist())+8) - 4.2*max((max(df_table.pontuacao.tolist())/10),1)]
                                    ))

        p.text(x='x_', y='y_', text='text', source=source_, text_font_size='14px', text_font_style='bold')

        escudo_url = df_clubes[df_clubes['id']==df_table.clube_id.tolist()[0]].escudo_url.tolist()[0]

        source_ = ColumnDataSource(dict(
                                url = [escudo_url],
                                x_  = [rodada_atual_num-8.6],
                                y_  = [max(10, max(df_table.pontuacao.tolist())+8) - 3.9*max((max(df_table.pontuacao.tolist())/10),1)]
                            ))

        image3 = ImageURL(url='url', x='x_', y='y_', w_units='screen', h_units='screen', w=40, h = 40, anchor="center")
        p.add_glyph(source_, image3)

    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.yaxis.axis_line_color= None
    p.outline_line_color = None
    p.yaxis.ticker = FixedTicker(ticks=[])
    p.xaxis.ticker = FixedTicker(ticks=[x for x in range(rodada_atual_num-7,max(df_table.rodada_id.tolist())+1)])
    p.legend.location = (max(10, max(df_table.pontuacao.tolist())), 430)
    p.legend.orientation = "horizontal"


    p.add_layout(Title(text='Últimos 7 Jogos de {0} \n Pontuação por Scout Positivo'.format(df_table.apelido.tolist()[0].upper()), align="center", text_font_size = '16px', text_font_style='italic'), "above")
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.major_label_text_font_style = 'bold'
    return p

#%%
lottie_url_hello = "https://assets3.lottiefiles.com/private_files/lf30_rg5wrsf4.json"
lottie_hello = load_lottieurl(lottie_url_hello)

stream.set_page_config(
     page_title="Cartologia | Cartola 2023",
     page_icon="🎲",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

c1, c2, c3, c4, c5, c6, c7, c8 = stream.columns([1, 1, 0.4, 3, 2, 0.5, 0.5, 0.5])

with c1:
    st_lottie(lottie_hello, key="hello", height=150,
        width=150)

c2.image('https://i.ibb.co/fvbCyHB/Imagem1.png', caption='Cartologia', width=150)

url_telegram = 'https://logodownload.org/wp-content/uploads/2017/11/telegram-logo.png'
url_insta = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/640px-Instagram_icon.png'
url_yt = 'https://logodownload.org/wp-content/uploads/2014/10/youtube-logo-5-2.png'

c3.image(url_telegram, width=60)
c3.image(url_insta, width=60)
c3.image(url_yt, width=60)


c4.write("##### Telegram")
c4.write("https://t.me/cartologia")
c4.write("##### Instagram")
c4.write("https://www.instagram.com/cartologiafc/")
c4.write("##### Youtube")
c4.write("https://www.youtube.com/channel/UCgwkcq7yeW1mOXdNXhszmOg")

c5.image('https://i.ibb.co/3CyZLwy/5371682.png', width=90)
c5.write("☕ Me compre um Café Cartológico")
c5.write("https://nubank.com.br/pagar/5jd11/73HqUsxHMi")

stream.write(" --- ")
stream.markdown("### 🎲 Cartologia by Thiago Villani | Soccer Analysis | Portal de Análises para Cartola FC ")
stream.markdown(" 💡 *Obs: Para otimizar visibilidade deixe o zoom do seu navegador em 75%*")

tabela_times, df_geodata_total, base_global_partidas, base_interacao_scouts, base_scouts, global_line_data_, df_confrontos_rodada_atual, df_atletas, df_data_scouts___, df_status, df_posicoes, base_global_partidas_og, rodada_atual_num, df_clubes, dataframe_player, depara_times_footstats, base_data_foodstat_, full_players_rank = get_full_datatables_required()

with stream.expander("Glossário & Premissas"):
        stream.write("""
            WebApp dedidcado para análises do Cartola FC 2023 \n
            Scouts = Eventos que ocorrem durantse uma partida de futebol \n
            A = Assistência - (Passes que resultam imediatamente em gol) \n
            G = Gols \n
            FS = Faltas sofridas \n
            FD = Finalizações Defendidados - (Chutem que vão no gol mas são defendidas) \n
            DS = Desarmes - (Admitidos com Interceptações declaradas + 50% de dividias vencidas) \n
            FF = Finalizações para Fora - (Chutes que não vão no gol) \n
            F =  Finalizações - (Representa a soma de FF + FD + FT) \n
        """)


stream.write(""" 
Selecione no menu abaixo uma das secções do Portal \n
                 Análise de Equipe -> Navege pelos principais indicadores de um clube específico \n
                 Análise de Campeonato -> Confira o TOP10 por scout e o Mapa do Confronto da Rodada \n
                 Sugestão de Escalação -> O Robô Cartológico tem o Algortimo para te ajudar a escalar o time da rodada
                """)
cb1, cb2, cb3, cb4 = stream.columns(4)
flag_plot_time_conteiner = cb1.checkbox('Análise de Equipe')
flag_plot_campeonato_conteiner = cb2.checkbox('Análise de Campeonato')
flag_plot_suggestion = False
flag_plot_pitch = cb3.checkbox('Sugestão de Escalação')
pos_selection = ""

stream.write("#### Rodada Atual - {0}".format(rodada_atual_num))
fig_lineup = get_basic_lineup()
stream.bokeh_chart(fig_lineup)


if(flag_plot_time_conteiner):
    list_of_teams = df_data_scouts___.clube_name.unique().tolist()

    cb1, cb2, cb3 = stream.columns(3)

    league_selection = cb1.selectbox(
    'Selecione um time',
        [""] + list_of_teams
    )

    player_selection = ""

    if(league_selection!=""):
        list_of_players = df_data_scouts___[df_data_scouts___['clube_name']==league_selection].apelido.unique().tolist()
        #player_selection = cb2.selectbox(
        #'Selecione um jogador',
        #[""] + list_of_players
        #)

        stream.markdown("#### Overview Geral")
        cb1, cb2 = stream.columns([1, 1])
        nome_footstats = tabela_times[tabela_times['clube_name']==league_selection].team_name.tolist()[0]
        fig_geral = get_pass_network_on_pitch_plot(nome_footstats, 'Passes', percentile_filter = 0.5, flag_filter_local = '', flag_insta_params = False, flag_hist=True)
        fig_geral.width = 900
        fig_geral.height = 600
        fig_geral.toolbar.logo = None
        fig_geral.toolbar_location = None
        fig_geral.xaxis.axis_line_color= None
        fig_geral.yaxis.axis_line_color= None
        fig_geral.outline_line_color = None
        fig_geral.background_fill_color = None
        fig_geral.border_fill_color = None
        cb1.bokeh_chart(fig_geral)

        f_round = get_round_plot_cartolafc(df_data_scouts___, df_clubes,  league_selection, min_num_jogos=0)
        f_round.width = 700
        f_round.height = 700
        cb2.bokeh_chart(f_round)

        stream.markdown("#### Históricos de Performance")
        cb1, cb2 = stream.columns([1, 1])
        clube_id = df_data_scouts___[df_data_scouts___['clube_name']==league_selection].clube_id.unique().tolist()[0]
        table_atuante = get_table_resumo_atuante(clube_id, flag_insta=False)
        table_atuante.width = 750
        table_atuante.height = 700
        cb1.bokeh_chart(table_atuante)

        clube_adv = df_confrontos_rodada_atual[df_confrontos_rodada_atual['clube_casa_id']==clube_id].clube_visitante_id.tolist()[0] if clube_id in df_confrontos_rodada_atual['clube_casa_id'].tolist() else  df_confrontos_rodada_atual[df_confrontos_rodada_atual['clube_visitante_id']==clube_id].clube_casa_id.tolist()[0]
        table_atuante = aux_func.get_table_resumo(clube_adv, df_clubes, df_data_scouts___, base_global_partidas_og, flag_insta=False)
        table_atuante.width = 750
        table_atuante.height = 700
        cb2.bokeh_chart(table_atuante)

        stream.write("----")

        stream.markdown("#### Radares de Scout para Confronto")
        cb1, cb2, cb3, cb4 = stream.columns([1, 1, 1, 1])
        f_radar = aux_func.get_radar_plot(clube_id, 'ata', df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
        f_radar.width = 400
        f_radar.height = 450
        cb1.bokeh_chart(f_radar)

        f_radar = aux_func.get_radar_plot(clube_id, 'lat', df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
        f_radar.width = 400
        f_radar.height = 450
        cb2.bokeh_chart(f_radar)

        f_radar = aux_func.get_radar_plot(clube_id, 'mei', df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
        f_radar.width = 400
        f_radar.height = 450
        cb3.bokeh_chart(f_radar)

        f_radar = aux_func.get_radar_plot(clube_id, 'zag', df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
        f_radar.width = 400
        f_radar.height = 450
        cb4.bokeh_chart(f_radar)

        stream.write("----")
        stream.markdown("#### TOP 5 Jogadores para Rodada")
        atleta_id_list = full_players_rank[full_players_rank['clube_name']==league_selection].head(5).atleta_id.tolist()
        
        for atleta_id in atleta_id_list:
            cb1, cb2, cb3 = stream.columns([1, 1, 1])
            f_round_quantile = aux_func.get_round_quantile_player(atleta_id, df_clubes, df_data_scouts___, dataframe_player)
            f_round_quantile.toolbar.logo = None
            f_round_quantile.toolbar_location = None
            cb1.bokeh_chart(f_round_quantile)

            f_ponts_dist = get_distribution_points_plot(df_data_scouts___, atleta_id)
            f_ponts_dist.width = 500
            f_ponts_dist.height = 500
            f_ponts_dist.toolbar.logo = None
            f_ponts_dist.toolbar_location = None
            cb2.bokeh_chart(f_ponts_dist)

            f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, atleta_id)
            f_points_scout.width = 500
            f_points_scout.height = 550
            f_points_scout.toolbar.logo = None
            f_points_scout.toolbar_location = None
            cb3.bokeh_chart(f_points_scout)

            stream.write("----")
        
        stream.markdown("#### Métricas de Moldelos Expected-Like  - G, npG, A")
        cb1, cb2, cb3 = stream.columns([1, 1, 1])
        f_xg = get_generic_horizontal_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xg',
                        title_suggested = 'Acc xGs',
                        agg_operation = 'sum',
                        flag_level_aggregation = 'apelido',
                        top_x = 7, flag_filter_position = '', flag_normalize_by_game=False, flag_filter_team=league_selection)

        f_xg.width = 500
        f_xg.height = 600
        cb1.bokeh_chart(f_xg)

        f_xg = get_generic_horizontal_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xA',
                        title_suggested = 'Acc xAs',
                        agg_operation = 'sum',
                        flag_level_aggregation = 'apelido',
                        top_x = 7, flag_filter_position = '', flag_normalize_by_game=False, flag_filter_team=league_selection)

        f_xg.width = 500
        f_xg.height = 600
        cb2.bokeh_chart(f_xg)

        f_xg = get_generic_horizontal_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'npxg',
                        title_suggested = 'Acc non-Penalty xG',
                        agg_operation = 'sum',
                        flag_level_aggregation = 'apelido',
                        top_x = 7, flag_filter_position = '', flag_normalize_by_game=False, flag_filter_team=league_selection)

        f_xg.width = 500
        f_xg.height = 600
        cb3.bokeh_chart(f_xg)

        


if(flag_plot_campeonato_conteiner):
    stream.write("----")
    stream.markdown("#### Mapas do Confronto")

    cb1, cb2 = stream.columns(2)
    f_mpconronto = aux_func.get_mapa_confronto_plot(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'ata', rodada_atual_num)
    f_mpconronto.toolbar.logo = None
    f_mpconronto.toolbar_location = None
    f_mpconronto.width = 700
    f_mpconronto.height = 700
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = aux_func.get_mapa_confronto_plot(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'mei', rodada_atual_num)
    f_mpconronto.toolbar.logo = None
    f_mpconronto.toolbar_location = None
    f_mpconronto.width = 700
    f_mpconronto.height = 700
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = aux_func.get_mapa_confronto_plot(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'lat', rodada_atual_num)
    f_mpconronto.toolbar.logo = None
    f_mpconronto.toolbar_location = None
    f_mpconronto.width = 700
    f_mpconronto.height = 700
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = aux_func.get_mapa_confronto_plot(df_data_scouts___, df_confrontos_rodada_atual, df_clubes, 'zag', rodada_atual_num)
    f_mpconronto.toolbar.logo = None
    f_mpconronto.toolbar_location = None
    f_mpconronto.width = 700
    f_mpconronto.height = 700
    cb2.bokeh_chart(f_mpconronto)
    stream.write("----")

    stream.markdown("#### Ranking Por Posições")

    stream.markdown("##### Ranking Zagueiros")
    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'DS',
        title_suggested = 'Desarmes',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'zag', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'FS',
        title_suggested = 'Faltas Sofridas',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'zag', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'DS',
        title_suggested = 'Desarmes',
        agg_operation = 'mean',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'zag', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'FS',
        title_suggested = 'Faltas Sofridas',
        agg_operation = 'mean',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'zag', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    stream.write("----")
    stream.markdown("##### Ranking Meias")
    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'F',
        title_suggested = 'Finalizações',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xA',
        title_suggested = 'xA',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'F',
        title_suggested = 'Finalizações',
        agg_operation = 'mean',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xA',
        title_suggested = 'xA',
        agg_operation = 'mean',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'DS',
        title_suggested = 'DS',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xg',
        title_suggested = 'npxg',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'DS',
        title_suggested = 'DS',
        agg_operation = 'mean',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xg',
        title_suggested = 'npxg',
        agg_operation = 'mean',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'mei', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    stream.write("----")
    stream.markdown("##### Ranking Laterais")
    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'F',
        title_suggested = 'Finalizações',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'lat', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xA',
        title_suggested = 'xA',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'lat', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'DS',
        title_suggested = 'DS',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'lat', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xg',
        title_suggested = 'xG',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'lat', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    stream.write("----")
    stream.markdown("##### Ranking Atacantes")
    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'F',
        title_suggested = 'Finalizações',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'ata', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xA',
        title_suggested = 'xA',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'ata', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)

    cb1, cb2 = stream.columns(2)
    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'DS',
        title_suggested = 'DS',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'ata', flag_filter_team = '')
    cb1.bokeh_chart(f_mpconronto)

    f_mpconronto = get_generic_vertical_bar(df_data_scouts___, global_line_data_, metric_to_plot = 'xg',
        title_suggested = 'xG',
        agg_operation = 'sum',
        flag_level_aggregation = 'apelido',
        top_x = 10, flag_filter_position = 'ata', flag_filter_team = '')
    cb2.bokeh_chart(f_mpconronto)


if(flag_plot_pitch):

    stream.write("----")
    stream.markdown("## Robô Cartológico")

    cb1, cb2 = stream.columns(2)

    cb1.write("""

           Selecione o fator de risco para pontuação cedida versus conquistada <br>
        Ex: Caso escolha 0 você considera APENAS o potencial de execução do time, caso escolha 1 você considera APENAS o potencial cedido pelo adversário. Recomenda-se algo na casa de 0.5 para escalações mais estáveis
        """)

    risk_factor = cb1.slider('(Fator de Risco (Alpha)', 0.0, 1.0, 0.5)

    cb1, cb2, cb3 = stream.columns(3)
    flag_seguro = cb1.checkbox('Proposta Segura')
    flag_moderado = cb2.checkbox('Proposta Moderada')
    flag_ousado = cb3.checkbox('Proposta Ousada')


    if(flag_seguro):
        stream.write("----")
        stream.markdown("##### Escalação - Segura")
        stream.write("""

            Segue abaixo os jogadores provavéis que compõe a equipe com os fatores de risco selecionados
        """)

        cb1, cb2, cb3 = stream.columns([1,2,1])
        type_ = 'seguro'
        fig_pitch_, list_players_seg = aux_func.get_pitch_top_player(df_atletas,
                                                                        df_data_scouts___,
                                                                        df_confrontos_rodada_atual,
                                                                        df_clubes,
                                                                            risk_factor,
                                                                            rodada_atual_num,
                                                                            ata_class = type_,
                                                                            mei_class = type_,
                                                                                lat_class = type_,
                                                                                zag_class = type_,
                                                                                gol_class = type_, min_num_jogos_ = int(rodada_atual_num/3))
        fig_pitch_.width = 700
        fig_pitch_.height = 500
        cb1.bokeh_chart(fig_pitch_)

        cb1, cb2, cb3, cb4 = stream.columns(4)
        flag_ata_seg = cb1.checkbox('Report - Análise de Atacantes | Seguros', value=True)
        flag_mei_seg = cb2.checkbox('Report - Análise de Meio Campo | Seguros', value=True)
        flag_lat_seg = cb3.checkbox('Report - Análise de Lateral | Seguros', value=True)
        flag_zag_seg = cb4.checkbox('Report - Análise de Zagueiro | Seguros', value=True)

        if(flag_ata_seg):
            with stream.expander("Atacantes"):
                stream.write("""
                    Relatório dos Atacantes
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

                
                
        
        if(flag_mei_seg):
            with stream.expander("Meio Campo"):
                stream.write("""
                    Relatório dos Meio Campo
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

        if(flag_lat_seg):
            with stream.expander("Laterais"):
                stream.write("""
                    Relatório dos Laterais
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)
                
                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)


        if(flag_zag_seg):
            with stream.expander("Zagueiros"):
                stream.write("""
                    Relatório dos Zagueiros
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2]):

                    f_ponts_dist = get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

    if(flag_moderado):
        stream.write("----")
        stream.markdown("##### Escalação - Moderada")
        stream.write("""

            Segue abaixo os jogadores provavéis que compõe a equipe com os fatores de risco selecionados
        """)

        cb1, cb2, cb3 = stream.columns([1,2,1])
        type_ = 'moderado'
        fig_pitch_, list_players_seg = aux_func.get_pitch_top_player(df_atletas,
                                                                        df_data_scouts___,
                                                                        df_confrontos_rodada_atual,
                                                                        df_clubes,
                                                                            risk_factor,
                                                                            rodada_atual_num,
                                                                            ata_class = type_,
                                                                            mei_class = type_,
                                                                                lat_class = type_,
                                                                                zag_class = type_,
                                                                                gol_class = type_, min_num_jogos_ = int(rodada_atual_num/3))
        fig_pitch_.width = 700
        fig_pitch_.height = 500
        cb1.bokeh_chart(fig_pitch_)

        cb1, cb2, cb3, cb4 = stream.columns(4)
        flag_ata_mod = cb1.checkbox('Report - Análise de Atacantes | Moderados', value=True)
        flag_mei_mod = cb2.checkbox('Report - Análise de Meio Campo | Moderados', value=True)
        flag_lat_mod = cb3.checkbox('Report - Análise de Lateral | Moderados', value=True)
        flag_zag_mod = cb4.checkbox('Report - Análise de Zagueiro | Moderados', value=True)

        if(flag_ata_mod):
            with stream.expander("Atacantes"):
                stream.write("""
                    Relatório dos Atacantes
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)
        
        if(flag_mei_mod):
            with stream.expander("Meio Campo"):
                stream.write("""
                    Relatório dos Meio Campo
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)
                
                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

        if(flag_lat_mod):
            with stream.expander("Laterais"):
                stream.write("""
                    Relatório dos Laterais
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)
                
                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

        if(flag_zag_mod):
            with stream.expander("Zagueiros"):
                stream.write("""
                    Relatório dos Zagueiros
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2]):

                    f_ponts_dist = get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

    if(flag_ousado):

        stream.write("----") 
        stream.markdown("##### Escalação - Ousado")
        stream.write("""

            Segue abaixo os jogadores provavéis que compõe a equipe com os fatores de risco selecionados
        """)

        cb1, cb2, cb3 = stream.columns([1,2,1])
        type_ = 'ousado'
        fig_pitch_, list_players_seg = aux_func.get_pitch_top_player(df_atletas,
                                                                        df_data_scouts___,
                                                                        df_confrontos_rodada_atual,
                                                                        df_clubes,
                                                                            risk_factor,
                                                                            rodada_atual_num,
                                                                            ata_class = type_,
                                                                            mei_class = type_,
                                                                                lat_class = type_,
                                                                                zag_class = type_,
                                                                                gol_class = type_, min_num_jogos_ = int(rodada_atual_num/3))
        fig_pitch_.width = 700
        fig_pitch_.height = 500
        cb1.bokeh_chart(fig_pitch_)

        cb1, cb2, cb3, cb4 = stream.columns(4)
        flag_ata_ous = cb1.checkbox('Report - Análise de Atacantes | Ousados', value=True)
        flag_mei_ous = cb2.checkbox('Report - Análise de Meio Campo | Ousados', value=True)
        flag_lat_ous = cb3.checkbox('Report - Análise de Lateral | Ousados', value=True)
        flag_zag_ous = cb4.checkbox('Report - Análise de Zagueiro | Ousados', value=True)

        if(flag_ata_ous):
            with stream.expander("Atacantes"):
                stream.write("""
                    Relatório dos Atacantes
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)
                
                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[:3],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)
        
        if(flag_mei_ous):
            with stream.expander("Meio Campo"):
                stream.write("""
                    Relatório dos Meio Campo
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[3:6],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

        if(flag_lat_ous):
            with stream.expander("Laterais"):
                stream.write("""
                    Relatório dos Laterais
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2]):

                    f_ponts_dist = aux_func.get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)
                
                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)
                
                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[6:8],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)

        if(flag_zag_ous):
            with stream.expander("Zagueiros"):
                stream.write("""
                    Relatório dos Zagueiros
                """)

                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):

                    f_round_quantile = aux_func.get_round_quantile_player(p_, df_clubes, df_data_scouts___, dataframe_player)
                    f_round_quantile.plot_width = 500
                    f_round_quantile.plot_height = 500
                    c__.bokeh_chart(f_round_quantile)
                
                cb1,cb2,cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2]):

                    f_ponts_dist = get_distribution_points_plot(df_data_scouts___, p_)
                    c__.bokeh_chart(f_ponts_dist)

                cb1, cb2, cb3 = stream.columns(3)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):
                    cid__ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).clube_id.tolist()[0]
                    pos_ = df_data_scouts___[df_data_scouts___['atleta_id']==p_].sort_values(by='rodada_id', ascending=False).posicao_nome.tolist()[0]
                    f_radar = aux_func.get_radar_plot(cid__, pos_, df_data_scouts___, df_confrontos_rodada_atual, df_clubes)
                    f_radar.plot_width = 450
                    f_radar.plot_height = 450
                    c__.bokeh_chart(f_radar)

                cb1, cb2 = stream.columns(2)
                for p_,c__ in zip(list_players_seg[8:10],[cb1,cb2,cb3]):
                    f_points_scout = get_plot_points_by_scout_breakdown(df_data_scouts___, p_)
                    f_points_scout.width = 500
                    f_points_scout.height = 550
                    f_points_scout.toolbar.logo = None
                    f_points_scout.toolbar_location = None
                    c__.bokeh_chart(f_points_scout)
