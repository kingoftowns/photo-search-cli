{{- define "photos-pv.fullname" -}}
photosearch-photos
{{- end }}

{{- define "photos-pv.labels" -}}
app.kubernetes.io/name: {{ include "photos-pv.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
