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

def add_image_in_collection(ee_project, asset_id, band, url_pattern, start_year, end_year, properties):
    
  for year in range(start_year, end_year+1):
      
    request = {
      'type': 'IMAGE',
      'properties': properties,
      'gcs_location': {
        'uris': [url_pattern.replace('{year}',str(year))]
      },
      'bands': { "id": band },
      'startTime': f'{year}-01-01T00:00:00.000000000Z',
      'endTime': f'{year}-12-31T23:59:59.000000000Z'
    }

    url = 'https://earthengine.googleapis.com/v1alpha/projects/{}/assets?assetId={}'

    asset_id_img = f"{asset_id}/{year}"
    print(f"Creating {asset_id_img}")
    
    response = session.post(
      url = url.format(ee_project, asset_id_img),
      data = json.dumps(request)
    )

ee_project = 'global-pasture-watch'

session = AuthorizedSession(
    ee.data.get_persistent_credentials().with_quota_project(ee_project)
)

cultiv_grassland_id = 'ggc-30m/v2/cultiv-grassland_p'
cultiv_grassland_band = 'probability'
cultiv_grassland_url = 'gs://wri-globalpasturewatch/ggc-30m/gpw_cultiv.grassland_rf.med.filt_p_30m_{year}0101_{year}1231_go_epsg.4326_v2.tif'

natsem_grassland_id = 'ggc-30m/v2/nat-semi-grassland_p'
natsem_grassland_band = 'probability'
natsem_grassland_url = 'gs://wri-globalpasturewatch/ggc-30m/gpw_nat.semi.grassland_rf.med.filt_p_30m_{year}0101_{year}1231_go_epsg.4326_v2.tif'

open_shrubland_id = 'ggc-30m/v2/open-shrubland_p'
open_shrubland_band = 'probability'
open_shrubland_url = 'gs://wri-globalpasturewatch/ggc-30m/gpw_open.shrubland_rf.med.filt_p_30m_{year}0101_{year}1231_go_epsg.4326_v2.tif'

domi_grassland_id = 'ggc-30m/v2/grassland_c'
domi_grassland_band = 'dominant_class'
domi_grassland_url = 'gs://wri-globalpasturewatch/ggc-30m/gpw_grassland_rf.med.filt.bthr_c_30m_{year}0101_{year}1231_go_epsg.4326_v2.tif'

start_time, end_time = '2000-01-01T00:00:00.000000000Z', '2024-12-31T23:59:59.000000000Z'
collection_properties = { 'version': 2.0 }


# The folder ggc-30m/v2 must exists
pprint(create_image_collection(ee_project, cultiv_grassland_id, start_time, end_time, collection_properties))
pprint(create_image_collection(ee_project, natsem_grassland_id, start_time, end_time, collection_properties))
pprint(create_image_collection(ee_project, open_shrubland_id, start_time, end_time, collection_properties))
pprint(create_image_collection(ee_project, domi_grassland_id, start_time, end_time, collection_properties))

add_image_in_collection(ee_project, cultiv_grassland_id, cultiv_grassland_band, cultiv_grassland_url, 2000, 2024, {})
add_image_in_collection(ee_project, natsem_grassland_id, natsem_grassland_band, natsem_grassland_url, 2000, 2024, {})
add_image_in_collection(ee_project, open_shrubland_id, open_shrubland_band, open_shrubland_url, 2000, 2024, {})
add_image_in_collection(ee_project, domi_grassland_id, domi_grassland_band, domi_grassland_url, 2000, 2024, {})