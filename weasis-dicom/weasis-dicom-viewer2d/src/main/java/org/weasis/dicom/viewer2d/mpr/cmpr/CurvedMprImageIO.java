/*
 * Copyright (c) 2024 Weasis Team and other contributors.
 *
 * This program and the accompanying materials are made available under the terms of the Eclipse
 * Public License 2.0 which is available at https://www.eclipse.org/legal/epl-2.0, or the Apache
 * License, Version 2.0 which is available at https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
 */
package org.weasis.dicom.viewer2d.mpr.cmpr;

import java.io.File;
import java.lang.ref.Reference;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import org.dcm4che3.data.Attributes;
import org.dcm4che3.data.SpecificCharacterSet;
import org.dcm4che3.data.Tag;
import org.dcm4che3.data.UID;
import org.dcm4che3.img.DicomMetaData;
import org.dcm4che3.util.UIDUtils;
import org.joml.Vector3d;
import org.opencv.core.CvType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.weasis.core.api.explorer.model.DataExplorerModel;
import org.weasis.core.api.media.data.Codec;
import org.weasis.core.api.media.data.FileCache;
import org.weasis.core.api.media.data.MediaElement;
import org.weasis.core.api.media.data.MediaSeriesGroup;
import org.weasis.core.api.media.data.TagW;
import org.weasis.core.util.SoftHashMap;
import org.weasis.dicom.codec.DcmMediaReader;
import org.weasis.dicom.codec.DicomImageElement;
import org.weasis.dicom.codec.DicomSeries;
import org.weasis.dicom.codec.TagD;
import org.weasis.dicom.codec.utils.DicomMediaUtils;
import org.weasis.dicom.viewer2d.mpr.Volume;
import org.weasis.opencv.data.ImageCV;
import org.weasis.opencv.data.PlanarImage;

/**
 * Image I/O handler for curved MPR panoramic images.
 * 
 * <p>This class generates the "straightened" panoramic view by sampling the volume along
 * the curved path defined in CurvedMprAxis.
 */
public class CurvedMprImageIO implements DcmMediaReader {
  private static final Logger LOGGER = LoggerFactory.getLogger(CurvedMprImageIO.class);

  private static final String MIME_TYPE = "image/cmpr";
  private static final SoftHashMap<CurvedMprImageIO, DicomMetaData> HEADER_CACHE =
      new SoftHashMap<>() {
        @Override
        public void removeElement(Reference<? extends DicomMetaData> soft) {
          CurvedMprImageIO key = reverseLookup.remove(soft);
          if (key != null) {
            hash.remove(key);
          }
        }
      };

  private final FileCache fileCache;
  private final HashMap<TagW, Object> tags;
  private final URI uri;
  private final CurvedMprAxis axis;
  private final Volume<?> volume;
  private Attributes attributes;

  public CurvedMprImageIO(CurvedMprAxis axis) {
    this.axis = Objects.requireNonNull(axis);
    this.volume = axis.getVolume();
    this.fileCache = new FileCache(this);
    this.tags = new HashMap<>();
    try {
      this.uri = new URI("data:" + MIME_TYPE);
    } catch (URISyntaxException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public void setBaseAttributes(Attributes attributes) {
    this.attributes = attributes;
  }

  @Override
  public PlanarImage getImageFragment(MediaElement media) throws Exception {
    return generatePanoramicImage();
  }

  /**
   * Generate the panoramic image by sampling along the curve.
   * 
   * <p>For a dental panoramic, we create a view where:
   * <ul>
   *   <li>Horizontal axis = distance along the curve (following the dental arch)</li>
   *   <li>Vertical axis = Z direction (superior-inferior, showing tooth height)</li>
   * </ul>
   * 
   * <p>At each (curve_position, z_level), we sample with a slab thickness perpendicular
   * to the curve and take the maximum intensity to capture the full tooth cross-section.
   */
  private PlanarImage generatePanoramicImage() {
    List<Vector3d> curvePoints = axis.getCurvePoints3D();
    if (curvePoints.size() < 2) {
      return null;
    }

    double stepMm = axis.getStepMm();
    double widthMm = axis.getWidthMm();
    double pixelMm = volume.getMinPixelRatio();
    Vector3d planeNormal = axis.getPlaneNormal();

    LOGGER.info("=== generatePanoramicImage DEBUG ===");
    LOGGER.info("curvePoints count: {}", curvePoints.size());
    LOGGER.info("First curve point: {}", curvePoints.get(0));
    LOGGER.info("Last curve point: {}", curvePoints.get(curvePoints.size() - 1));
    LOGGER.info("stepMm: {}, widthMm: {}, pixelMm: {}", stepMm, widthMm, pixelMm);
    LOGGER.info("planeNormal: {}", planeNormal);
    LOGGER.info("volume size: {}x{}x{}", volume.getSize().x, volume.getSize().y, volume.getSize().z);

    // Smooth the curve using Catmull-Rom spline interpolation to eliminate
    // sharp corners from rough user input
    List<Vector3d> smoothedPoints = smoothCurveWithSpline(curvePoints);
    
    List<Vector3d> sampledPoints = resampleCurve(smoothedPoints, stepMm, pixelMm);
    if (sampledPoints.isEmpty()) {
      return null;
    }
    
    // Reverse for dentist view (patient's right on viewer's left)
    Collections.reverse(sampledPoints);

    // Compute perpendicular directions at each sampled point (for slab thickness)
    List<Vector3d> perpDirs = computePerpendicularDirections(sampledPoints, planeNormal);

    LOGGER.info("sampledPoints count: {}", sampledPoints.size());
    if (!sampledPoints.isEmpty()) {
      LOGGER.info("First sampled point: {}", sampledPoints.get(0));
      LOGGER.info("Last sampled point: {}", sampledPoints.get(sampledPoints.size() - 1));
      LOGGER.info("First perpendicular dir: {}", perpDirs.get(0));
    }

    int widthPx = sampledPoints.size();
    
    // Height = Z extent to sample (widthMm controls vertical extent of panoramic)
    int heightPx = (int) Math.round(widthMm / pixelMm);
    if (heightPx < 1) heightPx = 1;
    
    // Slab thickness perpendicular to curve for MIP (in voxels)
    // Use a reasonable thickness to capture full tooth cross-section
    double slabThicknessMm = 20.0; // 20mm slab thickness
    int slabSamples = (int) Math.max(1, Math.round(slabThicknessMm / pixelMm));
    
    LOGGER.info("Output image: {}x{} (Z extent={}mm, slab={}mm, {} samples)", 
        widthPx, heightPx, widthMm, slabThicknessMm, slabSamples);

    int cvType = volume.getCVType();
    ImageCV dst = new ImageCV(heightPx, widthPx, cvType);

    // Get the reference Z level from curve (use average Z of curve points)
    double refZ = 0;
    for (Vector3d p : sampledPoints) {
      refZ += p.z;
    }
    refZ /= sampledPoints.size();
    LOGGER.info("Reference Z level: {}", refZ);

    // For each point along the curve, sample along Z with MIP perpendicular to curve
    // This creates a panoramic view showing tooth cross-sections
    for (int i = 0; i < widthPx; i++) {
      Vector3d P = sampledPoints.get(i);
      Vector3d perpDir = perpDirs.get(i);

      for (int j = 0; j < heightPx; j++) {
        // Z offset from reference (vertical axis of panoramic)
        double zOffset = (j - heightPx / 2.0);
        double sampleZ = refZ + zOffset;
        
        // Max intensity projection across the slab thickness perpendicular to curve
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < slabSamples; k++) {
          double perpOffset = (k - slabSamples / 2.0);
          double sampleX = P.x + perpDir.x * perpOffset;
          double sampleY = P.y + perpDir.y * perpOffset;
          
          Number value = volume.getInterpolatedValueFromSource(sampleX, sampleY, sampleZ);
          if (value != null && value.doubleValue() > maxValue) {
            maxValue = value.doubleValue();
          }
        }
        
        if (maxValue > Double.NEGATIVE_INFINITY) {
          setPixelValue(dst, j, i, maxValue, cvType);
        }
      }
    }
    
    LOGGER.info("Generated panoramic with Z sampling and perpendicular MIP slab");

    setDicomTags(widthPx, heightPx, pixelMm, stepMm);
    return dst;
  }

  /**
   * Compute the perpendicular direction to the curve tangent at each point.
   * The perpendicular is computed in the plane defined by planeNormal (typically XY plane).
   * This direction is used for the slab thickness in MIP.
   * 
   * @param sampledPoints the resampled curve points
   * @param planeNormal the normal of the source plane (e.g., Z for axial)
   * @return list of unit perpendicular directions at each point
   */
  private List<Vector3d> computePerpendicularDirections(
      List<Vector3d> sampledPoints, Vector3d planeNormal) {
    List<Vector3d> perpDirs = new ArrayList<>();
    int n = sampledPoints.size();
    
    for (int i = 0; i < n; i++) {
      Vector3d tangent;
      if (i == 0) {
        // Forward difference at start
        tangent = new Vector3d(sampledPoints.get(1)).sub(sampledPoints.get(0));
      } else if (i == n - 1) {
        // Backward difference at end
        tangent = new Vector3d(sampledPoints.get(n - 1)).sub(sampledPoints.get(n - 2));
      } else {
        // Central difference in the middle
        tangent = new Vector3d(sampledPoints.get(i + 1)).sub(sampledPoints.get(i - 1));
      }
      
      // Compute perpendicular in the plane: perp = planeNormal × tangent
      // This gives a vector perpendicular to both the tangent and the plane normal,
      // which lies in the plane and points "inward/outward" from the curve
      Vector3d perp = new Vector3d(planeNormal).cross(tangent);
      
      if (perp.lengthSquared() > 1e-10) {
        perp.normalize();
      } else {
        // Fallback if tangent is parallel to planeNormal (shouldn't happen for axial curves)
        perp = new Vector3d(1, 0, 0);
      }
      
      perpDirs.add(perp);
    }
    
    return perpDirs;
  }

  /**
   * Smooth the curve using Catmull-Rom spline interpolation.
   * This converts rough user-drawn polylines into smooth curves.
   * 
   * @param points the original control points
   * @return smoothed curve with many more points
   */
  private List<Vector3d> smoothCurveWithSpline(List<Vector3d> points) {
    if (points.size() < 2) return new ArrayList<>(points);
    if (points.size() == 2) return new ArrayList<>(points);
    
    List<Vector3d> result = new ArrayList<>();
    
    // Number of interpolated points between each pair of control points
    int segmentSamples = 20;
    
    for (int i = 0; i < points.size() - 1; i++) {
      // Get 4 control points for Catmull-Rom (with clamping at endpoints)
      Vector3d p0 = points.get(Math.max(0, i - 1));
      Vector3d p1 = points.get(i);
      Vector3d p2 = points.get(i + 1);
      Vector3d p3 = points.get(Math.min(points.size() - 1, i + 2));
      
      // Generate points along this segment
      for (int j = 0; j < segmentSamples; j++) {
        double t = (double) j / segmentSamples;
        Vector3d interpolated = catmullRom(p0, p1, p2, p3, t);
        result.add(interpolated);
      }
    }
    
    // Add the last point
    result.add(new Vector3d(points.get(points.size() - 1)));
    
    LOGGER.info("Smoothed curve: {} input points -> {} output points", 
        points.size(), result.size());
    
    return result;
  }
  
  /**
   * Catmull-Rom spline interpolation between p1 and p2.
   * 
   * @param p0 control point before p1
   * @param p1 start point of segment
   * @param p2 end point of segment  
   * @param p3 control point after p2
   * @param t interpolation parameter [0, 1]
   * @return interpolated point
   */
  private Vector3d catmullRom(Vector3d p0, Vector3d p1, Vector3d p2, Vector3d p3, double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    
    // Catmull-Rom basis functions
    double b0 = -0.5 * t3 + t2 - 0.5 * t;
    double b1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    double b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    double b3 = 0.5 * t3 - 0.5 * t2;
    
    return new Vector3d(
        b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x,
        b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y,
        b0 * p0.z + b1 * p1.z + b2 * p2.z + b3 * p3.z
    );
  }

  /**
   * Resample the curve to have evenly-spaced points.
   * Points are in voxel coordinates. We resample at 1-voxel intervals.
   */
  private List<Vector3d> resampleCurve(List<Vector3d> points, double stepMm, double pixelMm) {
    List<Vector3d> result = new ArrayList<>();
    if (points.size() < 2) return result;

    // Calculate total length in voxels
    double totalLengthVoxels = 0;
    for (int i = 1; i < points.size(); i++) {
      totalLengthVoxels += points.get(i).distance(points.get(i - 1));
    }

    if (totalLengthVoxels <= 0) return result;

    // Resample at 1-voxel step intervals for smooth output
    double stepVoxels = 1.0;
    int numSamples = (int) Math.ceil(totalLengthVoxels / stepVoxels);
    
    LOGGER.info("Resampling: totalLength={} voxels, numSamples={}", totalLengthVoxels, numSamples);
    
    for (int i = 0; i <= numSamples; i++) {
      double targetDist = i * stepVoxels;
      Vector3d point = interpolateAlongCurve(points, targetDist);
      if (point != null) {
        result.add(point);
      }
    }
    return result;
  }

  /**
   * Interpolate along the curve to find the point at a given distance (in voxels).
   */
  private Vector3d interpolateAlongCurve(List<Vector3d> points, double targetDistVoxels) {
    double accumulated = 0;
    for (int i = 1; i < points.size(); i++) {
      Vector3d p0 = points.get(i - 1);
      Vector3d p1 = points.get(i);
      double segmentLength = p0.distance(p1);
      if (accumulated + segmentLength >= targetDistVoxels) {
        double remaining = targetDistVoxels - accumulated;
        double t = segmentLength > 0 ? remaining / segmentLength : 0;
        return new Vector3d(p0).lerp(p1, t);
      }
      accumulated += segmentLength;
    }
    return points.isEmpty() ? null : new Vector3d(points.get(points.size() - 1));
  }

  private void setPixelValue(ImageCV dst, int row, int col, Number value, int cvType) {
    int depth = CvType.depth(cvType);
    switch (depth) {
      case CvType.CV_8U, CvType.CV_8S -> dst.put(row, col, value.byteValue());
      case CvType.CV_16U, CvType.CV_16S -> dst.put(row, col, value.shortValue());
      case CvType.CV_32S -> dst.put(row, col, value.intValue());
      case CvType.CV_32F -> dst.put(row, col, value.floatValue());
      case CvType.CV_64F -> dst.put(row, col, value.doubleValue());
    }
  }

  private void setDicomTags(int widthPx, int heightPx, double pixelMm, double stepMm) {
    HEADER_CACHE.remove(this);
    tags.put(TagD.get(Tag.Columns), widthPx);
    tags.put(TagD.get(Tag.Rows), heightPx);
    tags.put(TagD.get(Tag.SliceThickness), pixelMm);
    tags.put(TagD.get(Tag.PixelSpacing), new double[]{pixelMm, stepMm});
    tags.put(TagD.get(Tag.SOPInstanceUID), UIDUtils.createUID());
    tags.put(TagD.get(Tag.InstanceNumber), 1);
  }

  @Override
  public URI getUri() {
    return uri;
  }

  @Override
  public MediaElement getPreview() {
    return null;
  }

  @Override
  public boolean delegate(DataExplorerModel explorerModel) {
    return false;
  }

  @Override
  public DicomImageElement[] getMediaElement() {
    return null;
  }

  @Override
  public DicomSeries getMediaSeries() {
    return null;
  }

  @Override
  public int getMediaElementNumber() {
    return 1;
  }

  @Override
  public String getMediaFragmentMimeType() {
    return MIME_TYPE;
  }

  @Override
  public Map<TagW, Object> getMediaFragmentTags(Object key) {
    return tags;
  }

  @Override
  public void close() {
    HEADER_CACHE.remove(this);
  }

  @Override
  public Codec getCodec() {
    return null;
  }

  @Override
  public String[] getReaderDescription() {
    return new String[]{"Curved MPR Image Decoder"};
  }

  @Override
  public Object getTagValue(TagW tag) {
    return tag == null ? null : tags.get(tag);
  }

  @Override
  public void setTag(TagW tag, Object value) {
    DicomMediaUtils.setTag(tags, tag, value);
  }

  @Override
  public void setTagNoNull(TagW tag, Object value) {
    if (value != null) {
      setTag(tag, value);
    }
  }

  @Override
  public boolean containTagKey(TagW tag) {
    return tags.containsKey(tag);
  }

  @Override
  public Iterator<Entry<TagW, Object>> getTagEntrySetIterator() {
    return tags.entrySet().iterator();
  }

  public void copyTags(TagW[] tagList, MediaElement media, boolean allowNullValue) {
    if (tagList != null && media != null) {
      for (TagW tag : tagList) {
        Object value = media.getTagValue(tag);
        if (allowNullValue || value != null) {
          tags.put(tag, value);
        }
      }
    }
  }

  @Override
  public void replaceURI(URI uri) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Attributes getDicomObject() {
    Attributes dcm = new Attributes(tags.size() + (attributes != null ? attributes.size() : 0));
    if (attributes != null) {
      SpecificCharacterSet cs = attributes.getSpecificCharacterSet();
      dcm.setSpecificCharacterSet(cs.toCodes());
      dcm.addAll(attributes);
    }
    DicomMediaUtils.fillAttributes(tags, dcm);
    return dcm;
  }

  @Override
  public FileCache getFileCache() {
    return fileCache;
  }

  @Override
  public boolean buildFile(File output) {
    return false;
  }

  @Override
  public DicomMetaData getDicomMetaData() {
    return readMetaData();
  }

  @Override
  public boolean isEditableDicom() {
    return false;
  }

  @Override
  public void writeMetaData(MediaSeriesGroup group) {
    DcmMediaReader.super.writeMetaData(group);
  }

  private synchronized DicomMetaData readMetaData() {
    DicomMetaData header = HEADER_CACHE.get(this);
    if (header != null) {
      return header;
    }
    Attributes dcm = getDicomObject();
    header = new DicomMetaData(dcm, UID.ImplicitVRLittleEndian);
    org.dcm4che3.img.stream.ImageDescriptor desc = header.getImageDescriptor();
    if (desc != null) {
      org.opencv.core.Core.MinMaxLocResult minMax = new org.opencv.core.Core.MinMaxLocResult();
      minMax.minVal = volume.getMinimum();
      minMax.maxVal = volume.getMaximum();
      desc.setMinMaxPixelValue(0, minMax);
    }
    HEADER_CACHE.put(this, header);
    return header;
  }
}
