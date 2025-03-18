import os
import time
from datetime import timedelta

import numpy as np
from tqdm import tqdm

import rasterio
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box,Polygon
from scipy.spatial import ConvexHull

from core.util import raster_copy
from core.frame_info import image_normalize


def get_areas_and_polygons():
    """Read in the training rectangles and polygon shapefiles.

    Runs a spatial join on the two DBs, which assigns rectangle ids to all polygons in a column "index_right".
    """

    print("Reading training data shapefiles.. ", end="")
    start = time.time()

    # Read in areas and remove all columns except geometry
    areas = gpd.read_file(os.path.join(config.training_data_dir, config.training_area_fn))
    areas = areas.drop(columns=[c for c in areas.columns if c != "geometry"])

    # Read in polygons and remove all columns except geometry
    polygons = gpd.read_file(os.path.join(config.training_data_dir, config.training_polygon_fn))
    polygons = polygons.drop(columns=[c for c in polygons.columns if c != "geometry"])

    print(f"Done in {time.time()-start:.2f} seconds. Found {len(polygons)} polygons in {len(areas)} areas.\n"
          f"Assigning polygons to areas..      ", end="")
    start = time.time()

    # Perform a spatial join operation to pre-index all polygons with the rectangle they are in
    polygons = gpd.sjoin(polygons, areas, op="intersects", how="inner")

    print(f"Done in {time.time()-start:.2f} seconds.")
    return areas, polygons


def get_images_with_training_areas(areas):
    """Get a list of input images and the training areas they cover.

    Returns a list of tuples with img path and its area ids, eg [(<img_path>, [0, 12, 17, 18]), (...), ...]
    """
    
    print("Assigning areas to input images..  ")
    start = time.time()

    # Get all input image paths
    image_paths = []
    for root, dirs, files in os.walk(config.training_image_dir):
        for file in files:
            if file.startswith(config.train_image_prefix) and file.lower().endswith(config.train_image_type.lower()):
                image_paths.append(os.path.join(root, file))

    # Find the images that contain training areas
    images_with_areas = []
    for im in tqdm(image_paths):

        # Get image bounds
        with rasterio.open(im) as raster:
        #    im_bounds = box(*raster.bounds)
            mask = raster.read(1) > 0        #supposing valid values are strictly postive
            image_crs = raster.crs           #Used later for projeting area to same projection as input image
        coords = np.column_stack(np.where(mask))
    
        # Calculate the convex hull
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]
    
        # Create a polygon from the hull coordinates
        hull_polygon = Polygon([raster.transform*(x, y) for y, x in hull_coords])

        #gdf = gpd.GeoDataFrame(geometry=[box(*raster.bounds),hull_polygon])
        #gdf.to_file("my_file.shp")


        # Get training areas that are in this image
        #areas_in_image = np.where(areas.envelope.intersects(hull_polygon))[0]
        #areas_in_image = np.where(areas.envelope.within(hull_polygon))[0]
        areas_in_image = np.where(areas.to_crs(image_crs).envelope.within(hull_polygon))[0]
        

        if len(areas_in_image) > 0:
            images_with_areas.append((im, [int(x) for x in list(areas_in_image)], image_crs))

    print(f"Done in {time.time()-start:.2f} seconds. Found {len(image_paths)} training "
          f"images of which {len(images_with_areas)} contain training areas.")

    return images_with_areas


def calculate_boundary_weights(polygons, scale):
    """Find boundaries between close polygons.

    Scales up each polygon, then get overlaps by intersecting. The overlaps of the scaled polygons are the boundaries.
    Returns geopandas data frame with boundary polygons.
    """
    # Scale up all polygons around their center, until they start overlapping
    # NOTE: scale factor should be matched to resolution and type of forest
    scaled_polys = gpd.GeoDataFrame({"geometry": polygons.geometry.scale(xfact=scale, yfact=scale, origin='center')})

    # Get intersections of scaled polygons, which are the boundaries.
    boundaries = []
    for i in range(len(scaled_polys)):

        # For each scaled polygon, get all nearby scaled polygons that intersect with it
        nearby_polys = scaled_polys[scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)]

        # Add intersections of scaled polygon with nearby polygons [except the intersection with itself!]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                boundaries.append(scaled_polys.iloc[i].geometry.intersection(nearby_polys.iloc[j].geometry))

    # Convert to df and ensure we only return Polygons (sometimes it can be a Point, which breaks things)
    boundaries = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(boundaries)}).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # If we have boundaries, difference overlay them with original polygons to ensure boundaries don't cover labels
    if len(boundaries) > 0:
        print("test -----------------------------****************************")
        boundaries = gpd.overlay(boundaries, polygons, how='difference')
    else:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries


def preprocess_all(conf):
    """Run preprocessing for all training data."""

    global config
    config = conf

    print("Starting preprocessing.")
    start = time.time()

    # Create output folder
    output_dir = os.path.join(config.preprocessed_base_dir, time.strftime('%Y%m%d-%H%M')+'_'+config.preprocessed_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read in area and polygon shapefiles
    areas, polygons = get_areas_and_polygons()

    # Scan input images and find which images contain which training areas
    images_with_areas = get_images_with_training_areas(areas)

    # For each input image, get all training areas in the image
    for im_path, area_ids, image_crs in tqdm(images_with_areas, "Processing images with training areas", position=1):

        # For each area, extract the image channels and write img and annotation channels to a merged file
        for area_id in tqdm(area_ids, f"Extracting areas for {os.path.basename(im_path)}", position=0):
            areas=areas.to_crs(image_crs)
            # Extract the part of input image that overlaps training area, with optional resampling
            extract_ds = raster_copy("/vsimem/extracted", im_path, mode="translate", bounds=areas.bounds.iloc[area_id],
                                     resample=config.resample_factor, bands=list(config.preprocessing_bands + 1))

            # Create new  raster with two extra bands for labels and boundaries (float because we normalise im to float)
            n_bands = len(config.preprocessing_bands)
            mem_ds = gdal.GetDriverByName("MEM").Create("", xsize=extract_ds.RasterXSize, ysize=extract_ds.RasterYSize,
                                                        bands=n_bands + 2, eType=gdal.GDT_Float32)
            mem_ds.SetProjection(extract_ds.GetProjection())
            mem_ds.SetGeoTransform(extract_ds.GetGeoTransform())

            # Normalise image bands of the extract and write into new raster
            for i in range(1, n_bands+1):
                mem_ds.GetRasterBand(i).WriteArray(image_normalize(extract_ds.GetRasterBand(i).ReadAsArray()))

            # Write annotation polygons into second-last band       (GDAL only writes the polygons in the area bounds)
            #polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            #gdal.Rasterize(mem_ds, polygons_fp, bands=[n_bands+1], burnValues=[1], allTouched=config.rasterize_borders)

            # Write annotation polygons into the second-last band
            polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            polygons_2 = gpd.read_file(polygons_fp)

            # Reproject polygons to the image CRS
            polygons_2 = polygons.to_crs(image_crs)

            # Save the reprojected polygons to a temporary file
            temp_polygons_fp = "/vsimem/reprojected_polygons.shp"
            polygons_2.to_file(temp_polygons_fp)

            # Rasterize the reprojected polygons
            gdal.Rasterize(mem_ds, temp_polygons_fp, bands=[n_bands + 1], burnValues=[1], allTouched=config.rasterize_borders)

            

            # Get boundary weighting polygons for this area and write into last band
            polys_in_area = polygons[polygons.index_right == area_id]           # index_right was added in spatial join
            boundaries = calculate_boundary_weights(polys_in_area, scale=config.boundary_scale)
            if not boundaries.empty:
                boundaries.to_file("/vsimem/weights")
                gdal.Rasterize(mem_ds, "/vsimem/weights", bands=[n_bands+2], burnValues=[1], allTouched=True)
            else:
                print(f"Warning: No boundaries found for area {area_id}")
            #calculate_boundary_weights(polys_in_area, scale=config.boundary_scale).to_file("/vsimem/weights")
            
            #gdal.Rasterize(mem_ds, "/vsimem/weights", bands=[n_bands+2], burnValues=[1], allTouched=True)

            # Write extracted area to disk
            output_fp = os.path.join(output_dir, f"{area_id}.tif")
            gdal.GetDriverByName("GTiff").CreateCopy(output_fp, mem_ds, 0)

    if len(areas) > len(os.listdir(output_dir)):
        print(f"WARNING: Training images not found for {len(areas)-len(os.listdir(output_dir))} areas!")

    print(f"Preprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
