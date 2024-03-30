import os
import requests
import zipfile
import pandas as pd
import sqlite3 as sql

class Utils:

    @staticmethod
    def generate_triplets(db_path = 'data/database.db', csv_path = 'data/triplets.csv', t_tol = 12):

        conn = sql.connect(db_path)

        # Create a union query to get all the triplets
        sub_querys = [Utils.__query_question_q(q, t_tol) for q in range(1, 16)]
        union_query = " UNION ALL".join(sub_querys)
        query = f"""SELECT 
                        * 
                    FROM 
                        ({union_query}) as u 
                    WHERE 
                        respondent_id != 151945426 AND 
                        respondent_id != 815154444 AND 
                        respondent_id != 59 AND 
                        response IS NOT NULL
                    ORDER BY 
                        u.respondent_id, u.task_pos"""
        
        # Get the triplets and the places
        df_triplets = pd.read_sql_query(query, conn)
        required_columns = ['place_id', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5']
        df_places = pd.read_sql_query("SELECT * FROM Place", conn)[required_columns].copy()
        for img in [1,2,3,4,5]:
            df_places[f'image_{img}'] = df_places[f'image_{img}'].apply(lambda x: x.replace('assets/','data/'))

        # Merge the places with the triplets
        for i in range(1, 4):
            df_places_p = df_places.rename(columns=lambda x: x+f"_p{i}")
            df_triplets = pd.merge(df_triplets, df_places_p, left_on=f'place_{i}', right_on=f'place_id_p{i}', how='left')
            df_triplets = df_triplets.drop(columns=[f'place_id_p{i}'])
        
        # Create an index for the triplets and should be at the first column
        df_triplets['triplet_id'] = df_triplets.index
        df_triplets = df_triplets[['triplet_id'] + [col for col in df_triplets.columns if col != 'triplet_id']]

        # Save the triplets to a csv file
        df_triplets.to_csv(csv_path, index=False)


    @staticmethod
    def __query_question_q(q, t_tol):

        if q == 1:
            col_ti = 'start_survey'
            col_tf = 'next_button_1'
        else:
            col_ti = f'next_button_{q-1}'
            col_tf = f'next_button_{q}'

        return f"""
                SELECT
                    r.respondent_id, 
                    r.set_id, 
                    {q} as task_pos, 
                    s.task_{q} as taskid,
                    t.place_1 as place_1,
                    t.place_2 as place_2,
                    t.place_3 as place_3, 
                    r.response_{q} - 1 as response 
                FROM 
                    Response as r, 
                    Task_Set as s,
                    Task as t,
                    Timestamp as ts

                WHERE 
                    s.set_id = r.set_id AND
                    s.task_{q} = t.task_id AND
                    r.respondent_id = ts.respondent_id AND
                    ts.{col_tf} - ts.{col_ti} >= {t_tol}
                """


    @staticmethod
    def download_images(dest_path, url = "https://surfdrive.surf.nl/files/index.php/s/jo33wEPCfbOxvea/download"):
        print("Downloading images...")
        Utils.__download_image_zip(url, dest_path + '.zip')
        print("Download complete!")

        print("Unzipping images...")
        Utils.__unzip_images(dest_path + '.zip')
        print("Unzip complete!")

        print("Removing zip file...")
        os.remove('data/images.zip')

        print("Done!")


    @staticmethod
    def __download_image_zip(url, path_zipfile):
        r = requests.get(url, allow_redirects = True)
        open(path_zipfile, 'wb').write(r.content)


    @staticmethod
    def __unzip_images(path_zipfile):
        with zipfile.ZipFile(path_zipfile, 'r') as zip_ref:
            zip_ref.extractall('data/')

    ### FOR PLACES
            
    @staticmethod
    def generate_places_csv(pkl_file = 'places/Delft_NL.pkl', csv_file = 'places/Delft_NL_images.csv'):
        df = pd.read_pickle(pkl_file)
        df = df.reset_index(drop=True)
        df = df[['h3', 'image_path']].copy()
        im1 = df.groupby('h3').nth(0).set_index('h3').reset_index().rename(columns={'image_path': 'img_1'})
        im2 = df.groupby('h3').nth(1).set_index('h3').reset_index().rename(columns={'image_path': 'img_2'})
        im3 = df.groupby('h3').nth(2).set_index('h3').reset_index().rename(columns={'image_path': 'img_3'})
        im4 = df.groupby('h3').nth(3).set_index('h3').reset_index().rename(columns={'image_path': 'img_4'})
        im5 = df.groupby('h3').nth(4).set_index('h3').reset_index().rename(columns={'image_path': 'img_5'})
        imgs = pd.concat([im1, im2, im3, im4, im5], axis=1, ignore_index = False)
        imgs['h3_id'] = imgs['h3'].iloc[:, 0]
        imgs = imgs[['h3_id', 'img_1', 'img_2', 'img_3', 'img_4', 'img_5']].rename(columns={'h3_id': 'h3'})
        imgs.to_csv(csv_file, index=False)