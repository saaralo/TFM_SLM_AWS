import pandas as pd
from sklearn.naive_bayes import GaussianNB
import json

def clasificar(event, context):
    response = event['logs']
    df = pd.DataFrame.from_dict(response)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df[pd.to_datetime("today", utc=True) - df['fecha'] < pd.Timedelta(180, 'd')]
    df_alumnos = df[df['rol'] == 'alumno']
    alumno = df_alumnos[['tipo', 'ejercicio']].value_counts(dropna=False)
    al = alumno.to_frame().reset_index()
    alum = al[al['tipo'].isin(['finEjercicio', 'entraMundo', 'conejo', 'chat'])]
    alum['norma'] = alum[0]/alum[0].max()
    x_nick = preprocesar(alum)
    tipo = bayes(x_nick)
    tip = json.dumps({'tipo': tipo})
    resp ={
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type,application/json",
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        },
        "body": tip
    }
    return resp

def preprocesar(alum):
    x_nick = []
    # finEjercicio
    if(alum[alum['tipo'].isin(['finEjercicio'])].empty):
        x_nick.append(0)
    else:
        x_nick.append(0.9*alum[alum['tipo'] == 'finEjercicio'].groupby('tipo')['norma'].sum().to_list()[0])
        
    #entraMundo
    if(alum[alum['tipo'].isin(['entraMundo'])].empty):
        x_nick.append(0)
    else:
        x_nick.append(alum[alum['tipo'] == 'entraMundo']['norma'].to_list()[0])

    #conejo
    if(alum[alum['tipo'].isin(['conejo'])].empty):
        x_nick.append(0)
    else:
        x_nick.append(1.2*alum[alum['tipo'] == 'conejo']['norma'].to_list()[0])

    #chat
    if(alum[alum['tipo'].isin(['chat'])].empty):
        x_nick.append(0)
    else:
        x_nick.append(1.2*alum[alum['tipo'] == 'chat']['norma'].to_list()[0])

    #Graspme
    if(alum[alum['ejercicio'].isin(['Graspme'])].empty):
        x_nick.append(0)
    else:
        x_nick.append(2*alum[alum['ejercicio'] == 'Graspme']['norma'].to_list()[0])
    return x_nick

def bayes(x_nick):
    X = [[1,1,0,0,0],[1,0.5,0,0,0],[0,0.5,1,0,0],[0,0.5,0,1,1],[0,1,0,0,0]]
    y = ['Jugador','Triunfador','Espíritu Libre','Socializador','Filántropo']

    gnb = GaussianNB()
    gnb.fit(X, y)
    return gnb.predict([x_nick])[0]