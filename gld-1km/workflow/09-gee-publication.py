import ee
import json
from pprint import pprint
from google.auth.transport.requests import AuthorizedSession

def create_image_collection(ee_project, asset_id, start_time, end_time, properties):
    
    request = {
      'type': 'IMAGE_COLLECTION',
      'properties': properties,
      'startTime': start_time,
      'endTime': end_time,
    }

    
    url = 'https://earthengine.googleapis.com/v1alpha/projects/{}/assets?assetId={}'

    response = session.post(
      url = url.format(ee_project, asset_id),
      data = json.dumps(request)
    )

    return json.loads(response.content)

ee_project = 'global-pasture-watch'

session = AuthorizedSession(
    ee.data.get_persistent_credentials().with_quota_project(ee_project)
)

gld_id = 'gld-1km/v1/livestock-headcount-faostat_m'
start_time, end_time = '2000-01-01T00:00:00.000000000Z', '2022-12-31T23:59:59.000000000Z'
collection_properties = { 'version': 1.0 }
pprint(create_image_collection(ee_project, gld_id, start_time, end_time, collection_properties))

add_image_in_collection(ee_project, cultiv_grassland_id, cultiv_grassland_band, cultiv_grassland_url, 2000, 2022, {})
add_image_in_collection(ee_project, natsem_grassland_id, natsem_grassland_band, natsem_grassland_url, 2000, 2022, {})
add_image_in_collection(ee_project, domi_grassland_id, domi_grassland_band, domi_grassland_url, 2000, 2022, {})

y1,y2 = 2000, 2022

for year in range(y1,y2+1):

    request = {
      'imageManifest': {
        'name': f'projects/{ee_project}/assets/{gld_id}/{year}',
        'uriPrefix': 'gs://wri-globalpasturewatch/gld-1km/',
        'tilesets': [
          { 'id': '0', 'sources': [ { 'uris': [f"gpw_cattle.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"] } ]},
          { 'id': '1', 'sources': [ { 'uris': [f"gpw_buffalo.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"] } ]},
          { 'id': '2', 'sources': [ { 'uris': [f"gpw_horse.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"] } ]},
          { 'id': '3', 'sources': [ { 'uris': [f"gpw_goat.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"] } ]},
          { 'id': '4', 'sources': [ { 'uris': [f"gpw_sheep.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"] } ]}
        ],
        'bands': [
          { 'id': 'cattle', 'tilesetId': '0', 'tilesetBandIndex': 0 },
          { 'id': 'buffalo', 'tilesetId': '1', 'tilesetBandIndex': 0 },
          { 'id': 'horse', 'tilesetId': '2', 'tilesetBandIndex': 0 },
          { 'id': 'goat', 'tilesetId': '3', 'tilesetBandIndex': 0 },
          { 'id': 'sheep', 'tilesetId': '4', 'tilesetBandIndex': 0 },
        ],
        'startTime': f'{year}-01-01T00:00:00Z',
        'endTime': f'{year}-12-31T00:00:00Z'
      },
    }

    url = f'https://earthengine.googleapis.com/v1alpha/projects/{ee_project}/image:importExternal'

    print(f"Creating {request['imageManifest']['name']}")
    response = session.post(
      url = url,
      data = json.dumps(request)
    )
    
    pprint(json.loads(response.content))