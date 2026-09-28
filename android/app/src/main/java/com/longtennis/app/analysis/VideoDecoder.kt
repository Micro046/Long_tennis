package com.longtennis.app.analysis

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.Closeable
import kotlin.math.max
import kotlin.math.min

/** One decoded frame as packed BGR bytes (OpenCV layout) + its presentation time. */
class BgrFrame(val width: Int, val height: Int, val data: ByteArray, val index: Int, val ptsUs: Long) {
    inline fun b(x: Int, y: Int) = data[(y * width + x) * 3].toInt() and 255
    inline fun g(x: Int, y: Int) = data[(y * width + x) * 3 + 1].toInt() and 255
    inline fun r(x: Int, y: Int) = data[(y * width + x) * 3 + 2].toInt() and 255
}

data class VideoInfo(val width: Int, val height: Int, val durationUs: Long, val frameRate: Double, val rotation: Int, val estFrames: Int)

/**
 * Sequential MediaCodec decoder (ByteBuffer mode, YUV_420_888 -> BGR).
 * Frame timestamps come from the stream (variable frame rate safe).
 */
class VideoDecoder(private val context: Context, private val uri: Uri) : Closeable {
    private val extractor = MediaExtractor()
    private val codec: MediaCodec
    private val track: Int
    val info: VideoInfo
    private var inputDone = false
    private var outputDone = false
    private var nextIndex = 0

    init {
        extractor.setDataSource(context, uri, null)
        track = (0 until extractor.trackCount).first {
            extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        }
        extractor.selectTrack(track)
        val fmt = extractor.getTrackFormat(track)
        val w = fmt.getInteger(MediaFormat.KEY_WIDTH); val h = fmt.getInteger(MediaFormat.KEY_HEIGHT)
        val dur = if (fmt.containsKey(MediaFormat.KEY_DURATION)) fmt.getLong(MediaFormat.KEY_DURATION) else 0L
        val fr = if (fmt.containsKey(MediaFormat.KEY_FRAME_RATE)) fmt.getNumber(MediaFormat.KEY_FRAME_RATE)?.toDouble() ?: 30.0 else 30.0
        val rot = if (fmt.containsKey(MediaFormat.KEY_ROTATION)) fmt.getInteger(MediaFormat.KEY_ROTATION) else 0
        info = VideoInfo(w, h, dur, fr, rot, (dur / 1e6 * fr).toInt())
        codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME)!!)
        fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT, android.media.MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        codec.configure(fmt, null, null, 0)
        codec.start()
    }

    /** Returns the next frame or null at end of stream. */
    fun next(): BgrFrame? {
        val bi = MediaCodec.BufferInfo()
        while (!outputDone) {
            if (!inputDone) {
                val ii = codec.dequeueInputBuffer(10_000)
                if (ii >= 0) {
                    val buf = codec.getInputBuffer(ii)!!
                    val n = extractor.readSampleData(buf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true
                    } else {
                        codec.queueInputBuffer(ii, 0, n, extractor.sampleTime, 0); extractor.advance()
                    }
                }
            }
            val oi = codec.dequeueOutputBuffer(bi, 10_000)
            if (oi >= 0) {
                if (bi.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                if (bi.size > 0) {
                    val img = codec.getOutputImage(oi)
                    val frame = img?.let { toBgr(it, nextIndex, bi.presentationTimeUs) }
                    img?.close()
                    codec.releaseOutputBuffer(oi, false)
                    if (frame != null) { nextIndex++; return frame }
                } else codec.releaseOutputBuffer(oi, false)
            }
        }
        return null
    }

    private fun toBgr(img: android.media.Image, index: Int, pts: Long): BgrFrame {
        val w = img.cropRect.width(); val h = img.cropRect.height()
        val ox = img.cropRect.left; val oy = img.cropRect.top
        val py = img.planes[0]; val pu = img.planes[1]; val pv = img.planes[2]
        val yb = py.buffer; val ub = pu.buffer; val vb = pv.buffer
        val yRow = py.rowStride; val yPix = py.pixelStride
        val uRow = pu.rowStride; val uPix = pu.pixelStride
        val vRow = pv.rowStride; val vPix = pv.pixelStride
        val out = ByteArray(w * h * 3)
        val yArr = ByteArray(yb.remaining()); yb.get(yArr)
        val uArr = ByteArray(ub.remaining()); ub.get(uArr)
        val vArr = ByteArray(vb.remaining()); vb.get(vArr)
        var o = 0
        for (y in 0 until h) {
            val yy = y + oy
            val yBase = yy * yRow; val uvy = yy / 2
            for (x in 0 until w) {
                val xx = x + ox
                val Y = (yArr[yBase + xx * yPix].toInt() and 255) - 16
                val U = (uArr[uvy * uRow + (xx / 2) * uPix].toInt() and 255) - 128
                val V = (vArr[uvy * vRow + (xx / 2) * vPix].toInt() and 255) - 128
                val c = 1192 * max(0, Y)
                out[o++] = clamp((c + 2066 * U) shr 10)
                out[o++] = clamp((c - 400 * U - 833 * V) shr 10)
                out[o++] = clamp((c + 1634 * V) shr 10)
            }
        }
        return BgrFrame(w, h, out, index, pts)
    }

    private fun clamp(v: Int): Byte = min(255, max(0, v)).toByte()

    override fun close() {
        runCatching { codec.stop() }; runCatching { codec.release() }; runCatching { extractor.release() }
    }
}

/** Image resampling helpers on BGR frames (bilinear + area), mirroring cv2.resize. */
object Resample {
    /** Area-average downscale to (ow, oh), output planar BGR floats in [0,1] written at dst[offset + c*oh*ow + y*ow + x]. */
    fun areaToPlanar(f: BgrFrame, ow: Int, oh: Int, dst: FloatArray, offset: Int) {
        val sx = f.width.toDouble() / ow; val sy = f.height.toDouble() / oh
        val plane = ow * oh
        for (y in 0 until oh) {
            val y0 = (y * sy).toInt(); val y1 = min(f.height, max(y0 + 1, ((y + 1) * sy).toInt()))
            for (x in 0 until ow) {
                val x0 = (x * sx).toInt(); val x1 = min(f.width, max(x0 + 1, ((x + 1) * sx).toInt()))
                var b = 0; var g = 0; var r = 0
                for (yy in y0 until y1) {
                    var i = (yy * f.width + x0) * 3
                    for (xx in x0 until x1) {
                        b += f.data[i].toInt() and 255; g += f.data[i + 1].toInt() and 255; r += f.data[i + 2].toInt() and 255; i += 3
                    }
                }
                val n = ((y1 - y0) * (x1 - x0)).toFloat() * 255f
                val p = y * ow + x
                dst[offset + p] = b / n; dst[offset + plane + p] = g / n; dst[offset + 2 * plane + p] = r / n
            }
        }
    }

    /** Bilinear sample of channel c at (x, y) with border replicate. */
    fun bilinear(f: BgrFrame, x: Double, y: Double, c: Int): Float {
        val xf = min(f.width - 1.0, max(0.0, x)); val yf = min(f.height - 1.0, max(0.0, y))
        val x0 = xf.toInt(); val y0 = yf.toInt(); val x1 = min(f.width - 1, x0 + 1); val y1 = min(f.height - 1, y0 + 1)
        val ax = (xf - x0).toFloat(); val ay = (yf - y0).toFloat()
        fun px(xx: Int, yy: Int) = (f.data[(yy * f.width + xx) * 3 + c].toInt() and 255).toFloat()
        return (px(x0, y0) * (1 - ax) + px(x1, y0) * ax) * (1 - ay) + (px(x0, y1) * (1 - ax) + px(x1, y1) * ax) * ay
    }
}
