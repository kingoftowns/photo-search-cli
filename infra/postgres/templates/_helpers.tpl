{{- define "photosearch-postgres.fullname" -}}
photosearch-postgres
{{- end }}

{{- define "photosearch-postgres.labels" -}}
app.kubernetes.io/name: {{ include "photosearch-postgres.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "photosearch-postgres.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photosearch-postgres.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
