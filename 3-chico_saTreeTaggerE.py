# -*- coding: UTF-8 -*-
'''
Created on 06/11/2014

@author: larissa
'''

import csv
import glob
import gzip
import os
import pathlib
import pickle
import re
from os import chdir, makedirs, path, system
import pandas as pd
import json
import matplotlib.pyplot as plt
from  sklearn.metrics import accuracy_score, f1_score, fbeta_score, confusion_matrix, classification_report, balanced_accuracy_score

def leComentarios(comentarios):
    listaDeComentarios = []
    comentarios = open(comentarios,"r")
    for comentario in comentarios:
        identificador = comentario.partition(';')
        texto = identificador[2].partition('\n')
        listaDeComentarios = listaDeComentarios + [(identificador[0],texto[0])]
    #endfor
    return listaDeComentarios

def leComentariosPreProcessados(comentariosPreProcessados):
    listaDeComentariosPreProcessados = []
    # print('ler arquivo',comentariosPreProcessados)
    comentariosPreProcessados = open(comentariosPreProcessados,"r")
    # print('comentariosPreProcessados',comentariosPreProcessados.read())
    for comentarioPreProcessado in comentariosPreProcessados:
        # print('produzindo listaDeComentariosPreProcessados')
        palavra = comentarioPreProcessado.partition('\t')
        marcacao = palavra[2].partition('\t')
        lema = marcacao[2].partition('\n')
        listaDeComentariosPreProcessados = listaDeComentariosPreProcessados + [(palavra[0],marcacao[0],lema[0])]
    #endfor
    return listaDeComentariosPreProcessados

def leLexico(lexicoDeSentimento):
    listaDePalavraComPolaridade = []
    with open(lexicoDeSentimento) as lexico:
        palavrasComPolaridades = csv.reader(lexico, delimiter=',')
        for palavraComPolaridade in palavrasComPolaridades:
            listaDePalavraComPolaridade = listaDePalavraComPolaridade + [palavraComPolaridade]
        #endfor
    return listaDePalavraComPolaridade

def retornaComentariosPreProcessados(listaDeFeatures, listaDeComentariosPreProcessados):
    listaPorIdsDeComentarioPreProcessados = dict()
    indicesIds = []
    chaves = []
    indice = 0
    for tripla in listaDeComentariosPreProcessados:
        chave = re.findall("id_\d.*", tripla[0])
        if chave:
            # print(chave)
            indicesIds = indicesIds + [indice] #lista com as posicoes das chaves na listaDeComentariosPreProcessados [0,50,89,...]
            #print indicesIds
            chaves = chaves + chave
            #print chaves
        #endif
        indice = indice + 1
    #endfor
    for indiceId in range(0, len(indicesIds)):
        #print indiceId , " " , len(indicesIds)
        if (indiceId == len(indicesIds)-1):
            indicesIntervalo = range(indicesIds[indiceId]+2, len(listaDeComentariosPreProcessados))
        else:
            indicesIntervalo = range(indicesIds[indiceId]+2, indicesIds[indiceId+1])
        #endif
        triplasIntervalo = []
        for j in indicesIntervalo:
            triplasIntervalo = triplasIntervalo + [listaDeComentariosPreProcessados[j]]
        #endfor
        #print triplasIntervalo
        listaPorIdsDeComentarioPreProcessados[chaves[indiceId]] = triplasIntervalo
    #endfor
    return listaPorIdsDeComentarioPreProcessados

def regraAdjSentenca(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures):
    palavra = listaPorIdsDeComentarioPreProcessados[indiceChave][indiceComentarioPOS][2] #lema
    if feature.lower() == palavra.lower():
        antes = indiceComentarioPOS
        palavrasJanela = []
        for indicePre in range(0,indiceComentarioPOS):
            token = listaPorIdsDeComentarioPreProcessados[indiceChave][indicePre]
            if (token[1] == "Fp" or token[1] == "Fat" or token[1] == "Fs" or token[1] == "Fit"):
                palavrasJanela = []
                continue
            else:
                palavrasJanela = palavrasJanela + [token]
            #endif
        #endfor
        for indicePos in range(indiceComentarioPOS,len(listaPorIdsDeComentarioPreProcessados[indiceChave])-1):
            token = listaPorIdsDeComentarioPreProcessados[indiceChave][indicePos]
            if (token[1] == "Fp" or token[1] == "Fat" or token[1] == "Fs" or token[1] == "Fit"):
                break
            else:
                palavrasJanela = palavrasJanela + [token]
            #endif
        #endfor
        listaDeJanelasPorFeatures[feature] = listaDeJanelasPorFeatures[feature] + [(palavrasJanela,indiceChave)]
    #endif

def regraAdjJanelaNaSentenca(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures):
    palavra = listaPorIdsDeComentarioPreProcessados[indiceChave][indiceComentarioPOS][2] #lema
    if feature.lower() == palavra.lower():
        antes = indiceComentarioPOS - 3
        depois = indiceComentarioPOS + 3 + 1 #+1 por causa do range
        if (antes <= 0):
            antes = 0
        #endif
        if (depois >= len(listaPorIdsDeComentarioPreProcessados[indiceChave])):
            depois = len(listaPorIdsDeComentarioPreProcessados[indiceChave])
        #endif
        palavrasJanela = []
        for indiceJanela in range(antes, depois):
            token = listaPorIdsDeComentarioPreProcessados[indiceChave][indiceJanela]
            if indiceJanela < indiceComentarioPOS: #antes da feature
                if (token[1] == "Fp" or token[1] == "Fat" or token[1] == "Fs" or token[1] == "Fit"):
                    palavrasJanela = []
                    continue
                #endif
            elif indiceJanela > indiceComentarioPOS: #depois da feature
                if (token[1] == "Fp" or token[1] == "Fat" or token[1] == "Fs" or token[1] == "Fit"):
                    break
                #endif
            #endif
            palavrasJanela = palavrasJanela + [token]
        #endfor
        listaDeJanelasPorFeatures[feature] = listaDeJanelasPorFeatures[feature] + [(palavrasJanela,indiceChave)]
    #endif

def regraAdjJanela(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures):
    palavra = listaPorIdsDeComentarioPreProcessados[indiceChave][indiceComentarioPOS][2] #lema
    if feature.lower() == palavra.lower():
        antes = indiceComentarioPOS - 3
        depois = indiceComentarioPOS + 3 + 1 #+1 por causa do range
        if (antes <= 0):
            antes = 0
        #endif
        if (depois >= len(listaPorIdsDeComentarioPreProcessados[indiceChave])):
            depois = len(listaPorIdsDeComentarioPreProcessados[indiceChave])
        #endif
        palavrasJanela = []
        for indiceJanela in range(antes, depois):
            palavrasJanela = palavrasJanela + [listaPorIdsDeComentarioPreProcessados[indiceChave][indiceJanela]]
        #endfor
        listaDeJanelasPorFeatures[feature] = listaDeJanelasPorFeatures[feature] + [(palavrasJanela,indiceChave)]
    #endif

def regraAdjPrePos(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures,listaDeFeatures):
    palavra = listaPorIdsDeComentarioPreProcessados[indiceChave][indiceComentarioPOS][2] #lema
    #print feature.lower(), palavra.lower()
    if feature.lower() == palavra.lower():
        flagAdjAntes = 0
        antes = indiceComentarioPOS - 1
        if (antes < 0):
            antes = 0
        #endif
        categoria = listaPorIdsDeComentarioPreProcessados[indiceChave][antes][1]
        if ("AQ" in categoria):
            flagAdjAntes = 1
            depois = indiceComentarioPOS + 1 # +1 por causa do range, ADJ imediatamente antes da feature (ADJ anteposto)
        else:
            antes = indiceComentarioPOS
            depois = len(listaPorIdsDeComentarioPreProcessados[indiceChave])  # ADJ x casas depois da feature (ADJ posposto)
        #endif
        palavrasJanela = []
        for indiceJanela in range(antes, depois):
            palavraAtual = str(listaPorIdsDeComentarioPreProcessados[indiceChave][indiceJanela][2])
            marcacaoPalavraAtual = listaPorIdsDeComentarioPreProcessados[indiceChave][indiceJanela][1]
            #se achar outra feature, para
            if palavraAtual in listaDeFeatures and (indiceJanela!=antes and indiceJanela!=antes+1):
                break
            #endif
            #se acabou a sentenca, para
            if marcacaoPalavraAtual == "Fat" or marcacaoPalavraAtual == "Fp" or marcacaoPalavraAtual == "Fs" or marcacaoPalavraAtual == "Fit":
                break
            #endif
            palavrasJanela = palavrasJanela + [listaPorIdsDeComentarioPreProcessados[indiceChave][indiceJanela]] #salvando as palavras na janela
            if ("AQ" in marcacaoPalavraAtual) and flagAdjAntes==0:
                break
            #endif
        #endfor
        #print palavrasJanela
        listaDeJanelasPorFeatures[feature] = listaDeJanelasPorFeatures[feature] + [(palavrasJanela,indiceChave)]
    #endif

def retornaJanelasComFeatures(listaDeFeatures, listaPorIdsDeComentarioPreProcessados):
    listaDeJanelasPorFeatures = dict()
    for feature in listaDeFeatures:
        listaDeJanelasPorFeatures[feature] = []
    #endfor
    #somatorioTamanhoComentarios = 0
    #totalComentarios = len(listaPorIdsDeComentarioPreProcessados)
    for indiceChave in listaPorIdsDeComentarioPreProcessados:
        #print indiceChave
        #somatorioTamanhoComentarios = somatorioTamanhoComentarios + len(listaPorIdsDeComentarioPreProcessados[indiceChave])
        for indiceComentarioPOS in range(0,len(listaPorIdsDeComentarioPreProcessados[indiceChave])):
            #print indiceComentarioPOS
            for feature in listaDeFeatures:
                #regraAdjSentenca(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures)
                #regraAdjJanelaNaSentenca(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures)
                regraAdjJanela(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures)#era pra ser o melhor para positivo
                # regraAdjPrePos(feature,indiceComentarioPOS,listaPorIdsDeComentarioPreProcessados,indiceChave,listaDeJanelasPorFeatures,listaDeFeatures) # era pra ser o melhor para neg
            #enfor
        #endfor
    #endfor
    #print float(somatorioTamanhoComentarios)/float(totalComentarios) #tamanho medio dos comentarios
    return listaDeJanelasPorFeatures

def retornaJanelasComPolaridades(listaDePalavraComPolaridade, listaDeJanelasPorFeatures, inversaoPolaridade=True):
    # print(listaDeJanelasPorFeatures)
    listaDeJanelasComPolaridadeIntermediaria = []
    for janelas in listaDeJanelasPorFeatures.items():
        #print janelas
        numeroDeJanelas = len(janelas[1])
        #print numeroDeJanelas
        if numeroDeJanelas > 0:
            for janelaId in janelas[1]:
                #print janelaId
                texto = janelaId[0] #texto do documento
                id = janelaId[1] #id do documento
                polaridadeTrecho = 0
                for cadaPalavra in texto:
                    #print cadaPalavra
                    if ("AQ" in cadaPalavra[1]):
                        for polaridadeDaPalavra in listaDePalavraComPolaridade:
                            if (polaridadeDaPalavra[0] == cadaPalavra[2]): #funciona apenas para lista de adjetivos #lema
                                if inversaoPolaridade :
                                    polaridade = verificaInvercaoDePolaridade(texto, polaridadeDaPalavra[2])   #negacao
                                else:
                                    polaridade = polaridadeDaPalavra[2]
                                polaridadeTrecho = polaridadeTrecho + int(polaridade)
                                #print cadaPalavra[2],int(polaridade)
                            #endif
                        #endfor
                    #endif
                #endfor
                #print polaridadeTrecho,texto
                #trecho = ""
                #for i in texto:
                #    trecho = trecho + " " + i[0]
                #endfor
                #print trecho, id, cadaPalavra[0], polaridadeTrecho
                classe = janelas[0]
                temp1 = (classe, str(polaridadeTrecho), id)
                listaDeJanelasComPolaridadeIntermediaria.append(temp1)
            #endfor
        #endif
    #endfor
    listaPolaridadesFeaturesComentario = dict()
    for i in range(0,len(listaDeJanelasComPolaridadeIntermediaria)):
        featureAtual = str(listaDeJanelasComPolaridadeIntermediaria[i][0])
        polaridadeAtual = int(listaDeJanelasComPolaridadeIntermediaria[i][1])
        idAtual = str(listaDeJanelasComPolaridadeIntermediaria[i][2])
        if featureAtual in listaPolaridadesFeaturesComentario:
            tuplas = listaPolaridadesFeaturesComentario[featureAtual]
            if idAtual in tuplas:
                polaridadeNova = tuplas[idAtual] + polaridadeAtual
                tuplas[idAtual] = polaridadeNova
                listaPolaridadesFeaturesComentario[featureAtual]  = tuplas
            else:
                listaPolaridadesFeaturesComentario[featureAtual][idAtual] = polaridadeAtual
            #endif
        else:
            listaPolaridadesFeaturesComentario[featureAtual] = dict()
            listaPolaridadesFeaturesComentario[featureAtual][idAtual] = polaridadeAtual
        #endif
    #endfor
    # print ('pol, por feature ',listaPolaridadesFeaturesComentario)
    return listaPolaridadesFeaturesComentario
    #print listaDeJanelasComPolaridadeIntermediaria
    #return listaDejanelasComPolaridadeIntermediaria

def verificaInvercaoDePolaridade(janela, polaridade):
    listaDeNegacao = ["não","jamais","nada","nem","nenhum","ninguém","nunca","tampouco"]
    i=0
    for palavraMarcada in janela:
        palavra = palavraMarcada[0]
        categoria = palavraMarcada[1]
        lema = palavraMarcada[2]
        if lema in listaDeNegacao:
            if i < len(janela) - 1:
                #regra 1    NEG ADJ
                if("AQ" in janela[i+1][1]):
                    polaridade = -1*int(polaridade)
                    #print "inverteu regra 1"
                #endif
            #endif
            if i < len(janela) - 2:
                #regra 2    NEG VERBO ADJ
                if ("VM" in janela[i+1][1]) and ("AQ" in janela[i+2][1]):
                    polaridade = -1*int(polaridade)
                    #print "inverteu regra 2"
                #endif
            #endif
            if i < len(janela) - 3:
                #regra 3    NEG VERBO ADV ADJ
                if ("VM" in janela[i+1][1]) and ("RG" in janela[i+2][1]) and ("AQ" in janela[i+3][1]):
                    polaridade = -1*int(polaridade)
                    #print "inverteu regra 3"
                #endif
            #endif
            if i < len(janela) - 4:
                #regra 4    NEG VERBO PREP+DET ADJ
                if ("VM" in janela[i+1][1]) and ("SPC" in janela[i+2][1]) and ("DA" in janela[i+3][1]) and ("AQ" in janela[i+4][1]):
                    polaridade = -1*int(polaridade)
                    #print "inverteu regra 4"
                #endif
            #endif
        #endif
        i=i+1
    return polaridade

# def salvaComentariosComPolaridade(listaDeComentarios, listaPolaridadesFeaturesComentario, arquivoComMarcacaoAutomatica):
#     saida = open(arquivoComMarcacaoAutomatica,"w")
#     for classe in listaPolaridadesFeaturesComentario.keys():
#         for id in listaPolaridadesFeaturesComentario[classe].keys():
#             polaridade = listaPolaridadesFeaturesComentario[classe][id]
#             # print listaDeComentarios[0][0], id
#             for comentario in listaDeComentarios:
#                 if comentario[0].strip() == id.strip():
#                     comentarioAtual = comentario[1]
#                 #endif
#             #endfor
#             try:
#                 saida.write(str(comentarioAtual)+";"+str(classe)+";"+";"+";"+str(polaridade)+"\n")
#             except:
#                 # pass
#                 print 'erro  em ', id, comentario[0]
#         #endfor
#     #endfor
#     print "Salvando Arquivo"

def salvaComentariosComPolaridade(listaDeComentarios, listaPolaridadesFeaturesComentario, arquivoComMarcacaoAutomatica):
    # print (listaPolaridadesFeaturesComentario)
    # raise
    # print (arquivoComMarcacaoAutomatica)
    # print (listaDeComentarios)
    saida = []
    for classe in listaPolaridadesFeaturesComentario.keys():
        for id in listaPolaridadesFeaturesComentario[classe].keys():
            polaridade = listaPolaridadesFeaturesComentario[classe][id]
            # print (listaDeComentarios, id)
            for comentario in listaDeComentarios:
                if comentario[0].strip() == id.strip():
                    # print('comentario igual')
                    comentarioAtual = comentario[1]
                #endif
            #endfor
            saida.append([comentarioAtual,classe,None,None,polaridade])
        #endfor
    #endfor
    print (f"Salvando Arquivo com Marcação Automática:{arquivoComMarcacaoAutomatica}")
    # print (saida,arquivoComMarcacaoAutomatica)
    pd.DataFrame(saida).to_csv(arquivoComMarcacaoAutomatica,sep=';',header=None, index=False)

def precisao(vp, fp):
    try:
        calculo = vp/(vp + fp)
        return float(calculo)
    except ZeroDivisionError as detalhe:
        #print "Excecao:", detalhe
        return 0.0

def abrangencia(vp, fn):
    try:
        calculo = vp/(vp + fn)
        return float(calculo)
    except ZeroDivisionError as detalhe:
        #print "Excecao:", detalhe
        return 0.0

def acuracia(vp, vn, fp, fn):
    try:
        calculo = (vp + vn)/(vp + vn + fp + fn)
        return float(calculo)
    except ZeroDivisionError as detalhe:
        #print "Excecao:", detalhe
        return 0.0

def medidaF(vp, fp, fn):
    try:
        #print "vp ", vp, "fp ", fp, "fn ", fn
        calculo = (2*precisao(vp, fp)*abrangencia(vp, fn))/(precisao(vp, fp)+abrangencia(vp, fn))
        return float(calculo)
    except ZeroDivisionError as detalhe:
        #print "Excecao:", detalhe
        return 0.0

def get_metrics_tese_ulisses(arquivoMarcacaoManual, arquivoMarcacaoAutomatica, out_dir = 'output', binary_file ='Freitas_sem_neutro.csv', three_way_file = 'Freitas_com_neutro.csv', fold='fold_0', experiment = 'ontopt'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    marcacaoManual_df = pd.read_csv(arquivoMarcacaoManual, header=None, sep=';')
    # print(f'Abrindo marcacao manual {arquivoMarcacaoManual} ', '\n'*10)
    marcacaoManual_df.columns = ['review', 'feature', 'subfeature', 'subsubfeature', 'pol']
    marcacaoManual_df = marcacaoManual_df.drop(columns = ['subfeature', 'subsubfeature'])
    marcacaoManual_df.review = marcacaoManual_df.review.apply(lambda x: x.strip())
    marcacaoManual_df.feature = marcacaoManual_df.feature.apply(lambda x: x.strip())
    print(marcacaoManual_df.head(1))
    
    marcacaoAutomatica_df = pd.read_csv(arquivoMarcacaoAutomatica, header=None, sep=';')
    marcacaoAutomatica_df.columns = ['review', 'feature', 'subfeature', 'subsubfeature', 'pol']
    marcacaoAutomatica_df = marcacaoAutomatica_df.drop(columns = ['subfeature', 'subsubfeature'])
    marcacaoAutomatica_df.review = marcacaoAutomatica_df.review.apply(lambda x: x.strip())
    marcacaoAutomatica_df.feature = marcacaoAutomatica_df.feature.apply(lambda x: x.strip())
    print(marcacaoAutomatica_df.head(1))
    merged_df = marcacaoManual_df.merge(marcacaoAutomatica_df, how='inner', left_on=["review", "feature"], right_on=["review","feature"])
    merged_df.columns = ['review','feature', 'target', 'predict']
    # print("\n"*10,merged_df.target.unique(), "\n"*10)
    merged_df.predict[merged_df.predict < 0] = -1 
    merged_df_t = merged_df.copy()
    merged_df_t.predict[merged_df.predict > 0] = 1
    merged_df_t.predict[merged_df.predict == 0] = 0
    merged_df_t.drop_duplicates(inplace=True)
    
    merged_df_binary = merged_df.copy()
    merged_df_binary.predict[merged_df_binary.predict >= 0] = 1
    merged_df_binary.predict[merged_df_binary.predict < 0] = -1
    merged_df_binary = merged_df_binary[~(merged_df_binary.target == 0)]
    
    merged_df_binary.drop_duplicates(inplace=True)
    
    print('Binary\n',merged_df_binary.predict.unique())
    print('Ternary\n',merged_df)
    rep = pd.DataFrame(classification_report(y_true = merged_df_t.target, y_pred = merged_df_t.predict,output_dict=True)).transpose()
    print(rep)
    # acc = accuracy_score(y_true=merged_df_t.target, y_pred = merged_df_t.predict)
    # print(acc)
    # bacc = balanced_accuracy_score(y_true=merged_df_t.target, y_pred = merged_df_t.predict)
    # fscore = f1_score(y_true=merged_df_t.target, y_pred = merged_df_t.predict, average='macro')
    # three_way = {'acc': acc, 'bacc': bacc, 'fscore': fscore}
    # cm = confusion_matrix(merged_df_t.target, merged_df_t.predict)
    # print(cm)

    # ConfusionMatrixDisplay.from_predictions(merged_df_t.target, merged_df_t.predict).figure_.savefig(os.path.join(out_dir,'ternary_confusion_matrix.png'))
    # plt.show()
    rep_binary = pd.DataFrame(classification_report(y_true = merged_df_binary.target, y_pred = merged_df_binary.predict,output_dict=True)).transpose()
    print(rep_binary)
    # acc_bin = accuracy_score(y_true=merged_df_binary.target, y_pred = merged_df_binary.predict)
    # bacc_bin = balanced_accuracy_score(y_true=merged_df_binary.target, y_pred = merged_df_binary.predict)
    # fscore_bin = f1_score(y_true=merged_df_binary.target, y_pred = merged_df_binary.predict, average='macro')
    # two_way = {'acc': acc_bin, 'bacc': bacc_bin, 'fscore': fscore_bin}
    # print(acc_bin)
    # cm = confusion_matrix(merged_df_binary.target, merged_df_binary.predict)
    # print(cm)
    # ConfusionMatrixDisplay.from_predictions(merged_df_binary.target, merged_df_binary.predict).figure_.savefig(os.path.join(out_dir,'binary_confusion_matrix.png'))
    # plt.show()
    # out_dict = {'2-way':two_way, '3-way':three_way, 'fold': fold}

    merged_df_binary.to_csv(os.path.join(out_dir, binary_file), index = None, sep = ';')
    merged_df_t.to_csv(os.path.join(out_dir, three_way_file), index = None, sep = ';')
    
    # with open(os.path.join(out_dir,'metrics.json'), mode='w') as file:
    #     json.dump(out_dict, file, ensure_ascii=False)

def comparaResultados_old(marcacaoManual, marcacaoAutomatica, listaDeFeatures):
    pp_dic = dict()
    pn_dic = dict()
    pe_dic = dict() #neutro
    np_dic = dict()
    nn_dic = dict()
    ne_dic = dict() #neutro
    ep_dic = dict() #neutro
    en_dic = dict() #neutro
    ee_dic = dict() #neutro

    # listaDeFeatures = ["localização","quarto","atendimento","custo-benefício","limpeza"]

#ORDEM ALFABETICA
    # listaDeFeatures = ["aeroporto","apartamento","aparência","ar-condicionado","arena","atendimento","café da manhã","calefação","cama","casal","centro da cidade","chuveiro","cidade","colchão","conforto","corredor","cortina","cozinha","custo-benefício","ducha","elevador","escada","estabelecimento","estacionamento","frigobar","funcionários","gerente","gerência","horário","hotel","iluminação","instalações","internet","isolamento acústico","janta","lavanderia","lençóis","limpeza","localização","luxo","motel","móveis","padrão","país","pensão","portaria","porteiro","preço","quarto","quarto duplo","recepção","rua","serviço","serviço de quarto","shopping","tapete","telefone","televisão","toalha","tomada","torneira","travesseiro"]

#ORDEM FREQUENCIA
#    listaDeFeatures = ["quarto","hotel","café da manhã","localização","atendimento","preço","recepção","cama","chuveiro","elevador","funcionário","internet","serviço","instalação","televisão","lençol","toalha","limpeza","rua","estabelecimento","tapete","apartamento","corredor","ar-condicionado","custo-benefício","lavanderia","portaria","serviço de quarto","shopping","cozinha","frigobar","torneira","aeroporto","aparência","calefação","casal","ducha","estacionamento","horário","iluminação","isolamento acústico","janta","motel","padrão","telefone","tomada","pensão","arena","centro da cidade","cidade","colchão","conforto","cortina","escada","gerente","gerência","luxo","móvel","país","porteiro","quarto duplo","travesseiro"]

    for feature in listaDeFeatures:
        pp_dic[feature] = 0.0
        pn_dic[feature] = 0.0
        pe_dic[feature] = 0.0 #neutro
        np_dic[feature] = 0.0
        nn_dic[feature] = 0.0
        ne_dic[feature] = 0.0 #neutro
        ep_dic[feature] = 0.0 #neutro
        en_dic[feature] = 0.0 #neutro
        ee_dic[feature] = 0.0 #neutro
    #endfor
    for textoMarcacaoManual in marcacaoManual:
        comentarioMarcacaoManual = textoMarcacaoManual[0].lower().strip()
        featureMarcacaoManual = textoMarcacaoManual[1].lower().strip()
        for textoMarcacaoAutomatica in marcacaoAutomatica:
            comentarioMarcacaoAutomatica = textoMarcacaoAutomatica[0].lower().strip()
            featureMarcacaoAutomatica = textoMarcacaoAutomatica[1].lower().strip()
            if (comentarioMarcacaoManual == comentarioMarcacaoAutomatica and featureMarcacaoManual == featureMarcacaoAutomatica):
                for feature in listaDeFeatures:
                    pp_int = pp_dic[feature]
                    pn_int = pn_dic[feature]
                    pe_int = pe_dic[feature] #neutro
                    np_int = np_dic[feature]
                    nn_int = nn_dic[feature]
                    ne_int = ne_dic[feature] #neutro
                    ep_int = ep_dic[feature] #neutro
                    en_int = en_dic[feature] #neutro
                    ee_int = ee_dic[feature] #neutro
                    #print feature
                    if (textoMarcacaoManual[1].lower() == feature):
                        if(int(textoMarcacaoManual[4]) == 1):
                            if(int(textoMarcacaoAutomatica[4]) > 0):
                                pp_int = pp_int + 1
                                pp_dic[feature] = pp_int
                            #endif
                            if(int(textoMarcacaoAutomatica[4]) < 0):
                                pn_int = pn_int + 1
                                pn_dic[feature] = pn_int
                            #endif
                            if(int(textoMarcacaoAutomatica[4]) == 0): #neutro
                                pe_int = pe_int + 1
                                pe_dic[feature] = pe_int
                            #endif
                        #endif
                        if(int(textoMarcacaoManual[4]) == -1):
                            if(int(textoMarcacaoAutomatica[4]) > 0):
                                np_int = np_int + 1
                                np_dic[feature] = np_int
                            #endif
                            if(int(textoMarcacaoAutomatica[4]) < 0):
                                nn_int = nn_int + 1
                                nn_dic[feature] = nn_int
                            #endif
                            if(int(textoMarcacaoAutomatica[4]) == 0): #neutro
                                ne_int = ne_int + 1
                                ne_dic[feature] = ne_int
                            #endif
                        #endif
                        if(int(textoMarcacaoManual[4]) == 0): #neutro
                            if(int(textoMarcacaoAutomatica[4]) > 0):
                                ep_int = ep_int + 1
                                ep_dic[feature] = ep_int
                            #endif
                            if(int(textoMarcacaoAutomatica[4]) < 0):
                                en_int = en_int + 1
                                en_dic[feature] = en_int
                            #endif
                            if(int(textoMarcacaoAutomatica[4]) == 0):
                                ee_int = ee_int + 1
                                ee_dic[feature] = ee_int
                            #endif
                        #endif
                    #endif
                #endfor
            #endif
        #endfor
    #endfor
    precisao_p_dic = dict()
    abrangencia_p_dic = dict()
    acuracia_p_dic = dict()
    medidaF_p_dic = dict()
    precisao_n_dic = dict()
    abrangencia_n_dic = dict()
    acuracia_n_dic = dict()
    medidaF_n_dic = dict()
    precisao_e_dic = dict() #neutro
    abrangencia_e_dic = dict() #neutro
    acuracia_e_dic = dict() #neutro
    medidaF_e_dic = dict() #neutro
    print ("feature \t precisao_p_dic \t abrangencia_p_dic  \t medidaF_p_dic \t precisao_n_dic \t abrangencia_n_dic \t medidaF_n_dic")
    #positivo
    tot_vp = 0
    tot_fp = 0
    #negativo
    tot_vn = 0
    tot_fn = 0
    #neutro
    tot_ve = 0 
    tot_fe = 0
    for feature in listaDeFeatures:
        #print feature
        vp = pp_dic[feature]
        vn = nn_dic[feature] + ee_dic[feature]
        fp = np_dic[feature] + ep_dic[feature] #neutro
        fn = pn_dic[feature] + pe_dic[feature] #neutro
        
        #positivo
        tot_vp = tot_vp + vp
        tot_fp = tot_fp + fp
        #negativo
        tot_vn = tot_vn + vn
        tot_fn =  tot_fn + fn
        #neutro
        tot_ve = 0 
        tot_fe = 0
        
        precisao_p_dic = precisao(vp, fp)
        abrangencia_p_dic = abrangencia(vp, fn)
        acuracia_p_dic = acuracia(vp, vn, fp, fn) # REVISAR
        medidaF_p_dic = medidaF(vp, fp, fn)
        #print "Positivo"
        #print "P \t R \t A \t MF"
        #print "%1.2f" % precisao_p_dic + "\t" + "%1.2f" % abrangencia_p_dic + "\t" + "%1.2f" % acuracia_p_dic + "\t" + "%1.2f" % medidaF_p_dic
        #print "%1.2f" % medidaF_p_dic
        vp = nn_dic[feature]
        vn = pp_dic[feature] + ee_dic[feature]
        fp = pn_dic[feature] + en_dic[feature] #neutro
        fn = np_dic[feature] + ne_dic[feature] #neutro
        precisao_n_dic = precisao(vp, fp)
        abrangencia_n_dic = abrangencia(vp, fn)
        acuracia_n_dic = acuracia(vp, vn, fp, fn) # REVISAR
        medidaF_n_dic = medidaF(vp, fp, fn)
        #print "Negativo"
        #print "P \t R \t A \t MF"
        #print "%1.2f" % precisao_n_dic + "\t" + "%1.2f" % abrangencia_n_dic + "\t" + "%1.2f" % acuracia_n_dic + "\t" + "%1.2f" % medidaF_n_dic
        #print "%1.2f" % medidaF_n_dic
        vp = ee_dic[feature]
        vn = pp_dic[feature] + nn_dic[feature]
        fp = pe_dic[feature] + ne_dic[feature] #neutro
        fn = ep_dic[feature] + en_dic[feature] #neutro
        precisao_e_dic = precisao(vp, fp) #neutro
        abrangencia_e_dic = abrangencia(vp, fn) #neutro
        acuracia_e_dic = acuracia(vp, vn, fp, fn) #neutro # REVISAR
        medidaF_e_dic = medidaF(vp, fp, fn) #neutro
        #print "Neutro"
        #print "P \t R \t A \t MF"
        #print "%1.2f" % precisao_e_dic + "\t" + "%1.2f" % abrangencia_e_dic + "\t" + "%1.2f" % acuracia_e_dic + "\t" + "%1.2f" % medidaF_e_dic
#        print feature + "\t" + "%1.2f" % medidaF_p_dic + "\t" + "%1.2f" % medidaF_n_dic #+ "\t" + "%1.2f" % medidaF_e_dic
#        print "%1.2f" % medidaF_p_dic + "\t" + "%1.2f" % medidaF_n_dic #+ "\t" + "%1.2f" % medidaF_e_dic
        print (feature + "\t" + "%1.2f" % precisao_p_dic + "\t" + "%1.2f" % abrangencia_p_dic + "\t" + "%1.2f" % medidaF_p_dic + "\t" + "%1.2f" % precisao_n_dic + "\t" + "%1.2f" % abrangencia_n_dic + "\t" + "%1.2f" % medidaF_n_dic)
#         print "\t" + "%1.2f" % precisao_p_dic + "\t" + "%1.2f" % abrangencia_p_dic
#         print "\t" + "%1.2f" % precisao_n_dic + "\t" + "%1.2f" % abrangencia_n_dic
#         print "\t" + "%1.2f" % precisao_e_dic + "\t" + "%1.2f" % abrangencia_e_dic
    print ("\n")
    #endfor

def avaliaResultados(arquivoComMarcacaoManual, arquivoComMarcacaoAutomatica, listaDeFeatures):
    print ('avalia resultados')
    marcacaoManual = []
    with open(arquivoComMarcacaoManual) as manual:
        textosComMarcacaoManual = csv.reader(manual, delimiter=';')
        for textoComMarcacaoManual in textosComMarcacaoManual:
            marcacaoManual = marcacaoManual + [textoComMarcacaoManual]
        #endfor
    marcacaoAutomatica = []
    with open(arquivoComMarcacaoAutomatica) as automatica:
        textosComMarcacaoAutomatica = csv.reader(automatica,delimiter=';')
        for textoComMarcacaoAutomatica in textosComMarcacaoAutomatica:
            marcacaoAutomatica = marcacaoAutomatica + [textoComMarcacaoAutomatica]
        #endfor
    comparaResultados(marcacaoManual,marcacaoAutomatica, listaDeFeatures)

def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
    Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(object, file, protocol)
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = pickle.load(file)
    file.close()

    return object

def load_aspects_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def main():
    base_dir = 'Francisco/dataset_mestrado'
    print("Execution folder: ",os.getcwd())

    lexicos = ['AffectPT_br_editado', 'AffectPT_br', 'EmoLex', 'LeIA', 'LIWC', 'OntoPT', 'OpLexicon', 'ReLi_Lex', 'SentiLex', 'SentiWordNet', 'UNILEX', 'Wordnet_Affect_BR']

    lista_de_entradas = glob.glob(f"{base_dir}/lexicon/data/chico_hotelAll*.txt")
    for comentarios in lista_de_entradas:   
        variacao_entrada = comentarios.split('_')[-1].split('.')[0]
        listaDeFeatures = load_aspects_from_json(f'{base_dir}/lexicon/data/chico_aspectos{variacao_entrada}.json')
        listaDeComentarios = leComentarios(comentarios)
        comentariosPreProcessados = f"{base_dir}/lexicon/data/chico_tagged_hotel_{variacao_entrada}.txt"
        listaDeComentariosPreProcessados = leComentariosPreProcessados(comentariosPreProcessados)
        
        # print("pre proecessados ",listaDeComentariosPreProcessados) 
        # print('aqui')
        for lexico_experimento in lexicos:
            print(f'Experimento: {lexico_experimento}')
            
            lexicoDeSentimento = f"Lexicos - Dic/dic_{lexico_experimento}.csv"
            listaDePalavraComPolaridade = leLexico(lexicoDeSentimento) # funcionou a leitura do lexico

            out_dir = f"{base_dir}/output/{variacao_entrada}"
            out_dir = os.path.join(out_dir,lexico_experimento, 'fold_0')
            print(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)


            listaPorIdsDeComentarioPreProcessados = retornaComentariosPreProcessados(listaDeFeatures, listaDeComentariosPreProcessados)
            # print(listaPorIdsDeComentarioPreProcessados)
        
            listaDeJanelasPorFeatures = retornaJanelasComFeatures(listaDeFeatures, listaPorIdsDeComentarioPreProcessados)
            listaPolaridadesFeaturesComentario = retornaJanelasComPolaridades(listaDePalavraComPolaridade,listaDeJanelasPorFeatures, inversaoPolaridade=False)

            # print(listaPolaridadesFeaturesComentario)
            arquivoComMarcacaoManual = f'{base_dir}/lexicon/data/chico_AllFeaturesExplicitas_{variacao_entrada}.csv'
            
            arquivoComMarcacaoAutomatica = f"{out_dir}/chico_predicoes_freitas_treetagger_{variacao_entrada}_{lexico_experimento}.csv"
            # print(listaDeComentarios)
            listaDeComentariosComPolaridade = salvaComentariosComPolaridade(listaDeComentarios, listaPolaridadesFeaturesComentario, arquivoComMarcacaoAutomatica)
            # metricas = avaliaResultados(arquivoComMarcacaoManual, arquivoComMarcacaoAutomatica, listaDeFeatures)
            print("Arquivo Marcacao Automatica ", arquivoComMarcacaoAutomatica)
            get_metrics_tese_ulisses(arquivoComMarcacaoManual, arquivoComMarcacaoAutomatica,out_dir=out_dir, experiment=lexico_experimento)
    

if __name__ == "__main__":
    main() #Cross Validation
