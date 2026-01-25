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

    List<Vector3d> sampledPoints = resampleCurve(curvePoints, stepMm, pixelMm);
    if (sampledPoints.isEmpty()) {
      return null;
    }

    LOGGER.info("sampledPoints count: {}", sampledPoints.size());
    if (!sampledPoints.isEmpty()) {
      LOGGER.info("First sampled point: {}", sampledPoints.get(0));
      LOGGER.info("Last sampled point: {}", sampledPoints.get(sampledPoints.size() - 1));
    }

    int widthPx = sampledPoints.size();
    int heightPx = (int) Math.round(widthMm / pixelMm);
    if (heightPx < 1) heightPx = 1;

    LOGGER.info("Output image: {}x{}", widthPx, heightPx);

    int cvType = volume.getCVType();
    ImageCV dst = new ImageCV(heightPx, widthPx, cvType);

    // The sampling direction for the panoramic "height" should be ALONG the plane normal,
    // not perpendicular to it. For example, in axial view (normal = Z), we want to sample
    // up/down through the volume to see tooth roots.
    Vector3d samplingDir = new Vector3d(planeNormal);
    if (samplingDir.lengthSquared() < 1e-10) {
      samplingDir.set(0, 0, 1); // Default to Z if no normal provided
    } else {
      samplingDir.normalize();
    }

    for (int i = 0; i < widthPx; i++) {
      Vector3d P = sampledPoints.get(i);

      for (int j = 0; j < heightPx; j++) {
        double offsetMm = (j - heightPx / 2.0) * pixelMm;
        Vector3d offsetVox = new Vector3d(samplingDir).mul(offsetMm / pixelMm);
        Vector3d samplePoint = new Vector3d(P).add(offsetVox);
        Number value = volume.getInterpolatedValueFromSource(
            samplePoint.x, samplePoint.y, samplePoint.z);
        if (value != null) {
          setPixelValue(dst, j, i, value, cvType);
        }
      }
    }

    setDicomTags(widthPx, heightPx, pixelMm, stepMm);
    return dst;
  }

  private List<Vector3d> resampleCurve(List<Vector3d> points, double stepMm, double pixelMm) {
    List<Vector3d> result = new ArrayList<>();
    if (points.size() < 2) return result;

    double totalLength = 0;
    for (int i = 1; i < points.size(); i++) {
      totalLength += points.get(i).distance(points.get(i - 1)) * pixelMm;
    }

    if (totalLength <= 0) return result;

    int numSamples = (int) Math.ceil(totalLength / stepMm);
    for (int i = 0; i <= numSamples; i++) {
      double targetDist = i * stepMm;
      Vector3d point = interpolateAlongCurve(points, targetDist, pixelMm);
      if (point != null) {
        result.add(point);
      }
    }
    return result;
  }

  private Vector3d interpolateAlongCurve(List<Vector3d> points, double targetDistMm, double pixelMm) {
    double accumulated = 0;
    for (int i = 1; i < points.size(); i++) {
      Vector3d p0 = points.get(i - 1);
      Vector3d p1 = points.get(i);
      double segmentLength = p0.distance(p1) * pixelMm;
      if (accumulated + segmentLength >= targetDistMm) {
        double remaining = targetDistMm - accumulated;
        double t = segmentLength > 0 ? remaining / segmentLength : 0;
        return new Vector3d(p0).lerp(p1, t);
      }
      accumulated += segmentLength;
    }
    return points.isEmpty() ? null : new Vector3d(points.get(points.size() - 1));
  }

  private Vector3d computeTangent(List<Vector3d> points, int index) {
    Vector3d tangent = new Vector3d();
    if (points.size() < 2) {
      tangent.set(1, 0, 0);
      return tangent;
    }

    if (index == 0) {
      points.get(1).sub(points.get(0), tangent);
    } else if (index >= points.size() - 1) {
      points.get(points.size() - 1).sub(points.get(points.size() - 2), tangent);
    } else {
      points.get(index + 1).sub(points.get(index - 1), tangent);
    }

    if (tangent.lengthSquared() > 1e-10) {
      tangent.normalize();
    } else {
      tangent.set(1, 0, 0);
    }
    return tangent;
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
